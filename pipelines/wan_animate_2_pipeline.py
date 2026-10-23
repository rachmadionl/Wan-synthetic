import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import math
import inspect
import random
import subprocess
import time
import traceback
from datetime import timedelta
from functools import partial

import cv2
import numpy as np
from einops import rearrange
from loguru import logger
from decord import VideoReader
from easydict import EasyDict
from tqdm import tqdm

try:
    import moviepy.editor as mpy
except:
    import moviepy as mpy

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from core import build_object_from_dict, BaseInferencePipeline

from wanxiang import ops
from wanxiang.models.clip import AttentionBlock
from wanxiang.models.vae import VideoVAE
from wanxiang.utils.utils import TensorList, get_sharding_strategy, get_dtype
from wanxiang.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from wanxiang.eval_i2v import T5Encoder, CLIP, get_i2v_mask
from wanxiang.distributed.fsdp import shard_model, shard_transformer

from pipelines.utils.multiclip_utils import zigzag_padding, get_padding_len
from pipelines.utils.human_video_omni_utils import resize_by_area, get_frame_indices
from pipelines.utils.utils import ffmpeg_extract_audio_by_fps, merge_video_audio, get_video_duration
from pipelines.utils.color_utils import lab_mean, anchor_window_color, person_mask, person_stats, anchor_person, region_stats, anchor_regions
from safetensors import safe_open


def load_safetensors(path):
    tensors = {}
    with safe_open(path,  framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def merge_configs(main_cfg, backup_cfg):
    merged_cfg = EasyDict(main_cfg.copy())
    changes = []
    
    def _merge_dict(d1, d2, path=""):
        for key, value in d2.items():
            current_path = f"{path}.{key}" if path else key
            
            if key in d1:
                if isinstance(value, dict) and isinstance(d1[key], dict):
                    _merge_dict(d1[key], value, current_path)
                elif d1[key] != value:
                    changes.append(f"{current_path}: {d1[key]} -> {value}")
                    d1[key] = value
            else:
                d1[key] = value
                changes.append(f"新增 {current_path}: {value}")
    
    _merge_dict(merged_cfg, backup_cfg)
    
    if changes:
        logger.info("配置覆盖信息:")
        for change in changes:
            logger.info(f"  - {change}")
    else:
        logger.info("没有配置项被覆盖")
    
    return merged_cfg

def get_sampling_sigmas(sampling_steps, shift):
    sigma = np.linspace(1, 0, sampling_steps + 1)[:sampling_steps]
    sigma = shift * sigma / (1 + (shift - 1) * sigma)
    return sigma


def retrieve_timesteps(
    scheduler,
    num_inference_steps=None,
    device=None,
    timesteps=None,
    sigmas=None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def write_video(frames, path, fps, crf=10):
    """Encode RGB uint8 frames (T, H, W, 3) near-losslessly with libx264.

    moviepy's defaults produce ~0.5 Mbps at 720P, which bakes compression
    artifacts into the output before any upscaling. Odd dimensions (possible
    after padding removal) are cropped by one pixel for yuv420p.
    """
    frames = frames[:, : frames.shape[1] // 2 * 2, : frames.shape[2] // 2 * 2]
    t, h, w, _ = frames.shape
    proc = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
            "-c:v", "libx264", "-crf", str(crf), "-preset", "medium", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", path,
        ],
        stdin=subprocess.PIPE,
    )
    proc.stdin.write(np.ascontiguousarray(frames).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}")


# v4 (WAN_SPLIT_VAE=1): the per-window VAE work (encoding the continuity frame, decoding
# the window) is split into column strips, one per rank, instead of being repeated on
# every rank. Strips are computed with a halo of VAE_HALO latent columns, which covers
# the receptive field of the spatially-local VAE stages (outputs agree with the
# full-frame pass up to bf16 rounding, ~1/255 on a few percent of pixels).
VAE_HALO = 12


def _col_bounds(width, world):
    return [round(width * r / world) for r in range(world + 1)]


def _gather_cols(t, bounds):
    """all_gather the column strips [bounds[r], bounds[r+1]) of the last dim held by each rank."""
    wmax = max(b - a for a, b in zip(bounds, bounds[1:]))
    t = F.pad(t, (0, wmax - t.shape[-1])).contiguous()
    parts = [torch.empty_like(t) for _ in range(len(bounds) - 1)]
    dist.all_gather(parts, t)
    return torch.cat([p[..., : b - a] for p, a, b in zip(parts, bounds, bounds[1:])], -1)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class ForwardStepWrapper(nn.Module):
    def __init__(self, module):
        super(ForwardStepWrapper, self).__init__()
        self.wrapped_module = module

    def forward(self, *args, step=None, counter=None, **kwargs):
        return self.wrapped_module(*args, **kwargs)


def inference_core(item, gpu, cfg, clip, vae, model, t5, use_t5, context, context_null, device):
    height = item.height
    width = item.width
    CLIP_LEN = item.clip_len
    origin_len = CLIP_LEN
    origin_area = [width, height]

    # real_frame_len = item.real_frame_len
    
    first_num = item.first_num
    # first_img = item.first_img
    fps = item.fps
    seed = item.seed
    output_path = item.output_path
    # cond_images = item.cond_images

    prompt = item.prompt
    prompt_ref = item.prompt_ref
    neg_prompt = item.neg_prompt
    sample_guide_scale = item.sample_guide_scale
    step = item.step

    tpl_video_path = item.tpl_video_path
    refer_img_path = item.refer_img_path

    cfg.test_cfg.sample_guide_scale = sample_guide_scale

    setup_seed(seed)
    seed_g = torch.Generator(device=gpu)
    seed_g.manual_seed(seed)
    
    start = 0
    end = CLIP_LEN
    all_out_frames = []

    logger.info(f"Processing reference image: {refer_img_path}")
    first_img = cv2.imread(refer_img_path)[..., ::-1]
    first_img, pading_info = resize_by_area(first_img, width * height, divisor=16, return_padding_info=True)


    # 读取视频
    video_reader = VideoReader(tpl_video_path)
    frame_num = len(video_reader)
    logger.info('frame_num: {}'.format(frame_num))

    # fps采样，短的首尾循环
    video_fps = video_reader.get_avg_fps()
    logger.info(f"[GPU {gpu}] video fps {video_fps}")

    audio_save_path = os.path.join(output_path, 'tpl_audio.wav')
    ffmpeg_extract_audio_by_fps(tpl_video_path, audio_save_path, fps)
    
    # TODO: Maybe we can switch to PyAV later, which can get accurate frame num
    duration = float(video_reader.get_frame_timestamp(-1)[-1])
    if not (np.isfinite(duration) and 0 < duration < frame_num / video_fps * 10):
        logger.warning(f"[GPU {gpu}] invalid duration {duration} from decord, fallback to ffprobe")
        duration = get_video_duration(tpl_video_path)
        logger.info(f"[GPU {gpu}] duration (ffprobe): {duration}")
    expected_frame_num = int(duration * video_fps + 0.5)
    ratio = abs((frame_num - expected_frame_num) / frame_num)
    if ratio > 0.1:
        logger.warning(f"[GPU {gpu}] Warning: The difference between the actual number of frames and the expected number of frames is too large")
        frame_num = expected_frame_num

    target_num = int(frame_num / video_fps * fps)
    idxs = get_frame_indices(frame_num, video_fps, target_num, fps)

    frames = video_reader.get_batch(idxs).asnumpy()
    split_vae = os.environ.get("WAN_SPLIT_VAE") == "1" and dist.get_world_size() > 1
    if split_vae:
        # resizing 400 1080p frames takes ~15 s on CPU; each rank resizes a share and the
        # results are exchanged over NCCL
        world, rank = dist.get_world_size(), dist.get_rank()
        per = -(-len(frames) // world)
        mine = [resize_by_area(f, width * height, divisor=16) for f in frames[rank * per:(rank + 1) * per]]
        shape = resize_by_area(frames[0], width * height, divisor=16).shape
        buf = torch.zeros(per, *shape, dtype=torch.uint8, device=device)
        if mine:
            buf[:len(mine)] = torch.from_numpy(np.stack(mine)).to(device)
        parts = [torch.empty_like(buf) for _ in range(world)]
        dist.all_gather(parts, buf)
        cond_images = list(torch.cat(parts)[:len(frames)].cpu().numpy())
        del buf, parts, mine
    else:
        cond_images = [resize_by_area(frame, width * height, divisor=16) for frame in frames]
    real_frame_len = len(frames)

    logger.info('real_frame_len: {}'.format(real_frame_len))
    target_len = get_padding_len(real_frame_len, CLIP_LEN, first_num)

    logger.info('target_len: {}'.format(target_len))
    cond_images = zigzag_padding(cond_images, target_len)

    # v3 (WAN_PRECOMPUTE=1): encode the driving video once up front, windows split
    # round-robin across ranks (the VAE is replicated, so ranks encode in parallel)
    # and broadcast, instead of every rank re-encoding every window. Per-video constants
    # (reference latent/CLIP features, T5 prompts) are computed once in the first window.
    precompute = os.environ.get("WAN_PRECOMPUTE") == "1"
    cond_cache, const_cache = {}, {}
    if precompute:
        windows, s_, e_, clip_ = [], 0, CLIP_LEN, CLIP_LEN
        while s_ + first_num < len(cond_images):
            if len(cond_images) - s_ < clip_:
                clip_ = len(cond_images) - s_
            windows.append((s_, e_))
            s_ += clip_ - first_num
            e_ += clip_ - first_num
        world, rank = dist.get_world_size(), dist.get_rank()
        ch, cw = cond_images[0].shape[:2]
        torch.cuda.synchronize(); t_pre = time.time()
        with torch.autocast(device_type=str(device), dtype=torch.bfloat16), torch.no_grad():
            for w, (s_, e_) in enumerate(windows):
                n = len(cond_images[s_:e_])
                if w % world == rank:
                    x = rearrange(torch.tensor(np.stack(cond_images[s_:e_]) / 127.5 - 1),
                                  "t h w c -> 1 c t h w").to(device=device, dtype=torch.bfloat16)
                    cond_cache[w] = torch.stack(vae.encode(x)).float().contiguous()
                    del x
                else:
                    cond_cache[w] = torch.empty(1, 16, (n - 1) // 4 + 1, ch // 8, cw // 8,
                                                device=device, dtype=torch.float32)
        for w in range(len(windows)):
            dist.broadcast(cond_cache[w], src=w % world)
        torch.cuda.synchronize()
        logger.info(f"[GPU {gpu}] precomputed {len(windows)} driving-video windows in {time.time() - t_pre:.1f}s")

    torch.cuda.empty_cache()
    idex = 0
    while True:
        if start + first_num >= len(cond_images):
            break
        
        if start == 0:
            mask_reft_len = 0
        else:
            mask_reft_len = first_num

        if len(cond_images) - start < CLIP_LEN:
            CLIP_LEN = len(cond_images) - start
            logger.info(f"[GPU {gpu}] Start: {start} End: {end} CLIP_LEN: {CLIP_LEN}")
            
        # process the input
        batch = {
            "conditioning_pixel_values": torch.zeros(1, 3, CLIP_LEN, height, width),
            "refer_pixel_values": torch.zeros(1, 3, height, width),
            "refer_t_pixel_values": torch.zeros(first_num, 3, height, width),
        }
       
        if precompute:
            # the window's latents are precomputed; only its first frame is still used (CLIP)
            batch["conditioning_pixel_values"] = rearrange(
                torch.tensor(cond_images[start][None] / 127.5 - 1),
                "t h w c -> 1 c t h w",
            )
        else:
            batch["conditioning_pixel_values"] = rearrange(
                torch.tensor(np.stack(cond_images[start:end]) / 127.5 - 1),
                "t h w c -> 1 c t h w",
            )

        batch["refer_pixel_values"] = rearrange(
            torch.tensor(first_img / 127.5 - 1), "h w c -> 1 c h w"
        )

        if start > 0:
            batch["refer_t_pixel_values"] = rearrange(
                out_frames[0, :, -first_num:].clone().detach(),
                "c t h w -> t c h w",
            )

        batch['pixel_values'] = rearrange(batch["refer_t_pixel_values"],
                                          "t c h w -> 1 c t h w",
                                          )
        idex += 1
        dist.barrier(device_ids=[gpu])
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device=device, dtype=torch.bfloat16)

        ref_pixel_values = batch["refer_pixel_values"]
        conditioning_pixel_values = batch["conditioning_pixel_values"]

        B, C, H, W = ref_pixel_values.shape

        T = CLIP_LEN + 1
        lat_h = H // 8
        lat_w = W // 8
        lat_t = T // 4 + 1 + 1
        target_shape = [lat_t, lat_h, lat_w]
        grid_sizes = torch.stack([torch.tensor([lat_t, lat_h // 2, lat_w // 2], dtype=torch.long)])

        noise = [
            torch.randn(
                16,
                target_shape[0],
                target_shape[1],
                target_shape[2],
                dtype=torch.float32,
                device=device,
                generator=seed_g,
            )
        ]

        torch.cuda.empty_cache()
        torch.cuda.synchronize(); t_window = time.time()

        with (
            torch.autocast(device_type=str(device), dtype=torch.bfloat16, enabled=True),
            torch.no_grad()
        ):  
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=cfg.test_cfg.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sampling_sigmas = get_sampling_sigmas(step, cfg.test_cfg.sample_shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler, device=device, sigmas=sampling_sigmas
            )
            latents = noise

            ref_pixel_values = rearrange(ref_pixel_values, "t c h w -> 1 c t h w")
            if "ref_latents" in const_cache:
                ref_latents = const_cache["ref_latents"]
            else:
                ref_latents = torch.stack(vae.encode(ref_pixel_values.to(torch.bfloat16)))
                if precompute:
                    const_cache["ref_latents"] = ref_latents
            mask_ref = get_i2v_mask(1, lat_h, lat_w, 1, device=device)
            y_ref = torch.concat([mask_ref, ref_latents[0]]).to(dtype=torch.bfloat16, device=device)

            if os.environ.get("WAN_MEMLOG") and gpu == 0:
                print(f"[memlog] after_ref_vae: allocated {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
            img = ref_pixel_values[0, :, 0]
            if "clip_context" in const_cache:
                clip_context = const_cache["clip_context"]
            else:
                clip_context = clip.visual([img[:, None, :, :]]).to(dtype=torch.bfloat16, device=device)
                if precompute:
                    const_cache["clip_context"] = clip_context

            msk_reft = get_i2v_mask(lat_t - 1, lat_h, lat_w, mask_reft_len, device=device)

            if mask_reft_len > 0:
                y_reft_in = torch.concat(
                    [
                        torch.nn.functional.interpolate(batch["pixel_values"][0, :, :mask_reft_len].cpu(),
                                                        size=(H, W), mode="bicubic"),
                        torch.zeros(3, T - mask_reft_len - 1, H, W),
                    ],
                    dim=1,
                ).to(device)
            else:
                y_reft_in = torch.concat(
                    [
                        torch.zeros(3, T - mask_reft_len - 1, H, W),
                    ],
                    dim=1,
                ).to(device)
            if split_vae:
                world, rank = dist.get_world_size(), dist.get_rank()
                bounds = _col_bounds(W // 8, world)
                y_reft = vae.encode_cols(y_reft_in, bounds[rank], bounds[rank + 1],
                                         lambda h: _gather_cols(h, bounds), halo=VAE_HALO)
            else:
                y_reft = vae.encode([y_reft_in])[0]
            del y_reft_in

            y_reft = torch.concat([msk_reft, y_reft]).to(dtype=torch.bfloat16, device=device)
            y = torch.concat([y_ref, y_reft], dim=1)

            if precompute:
                condition_latents = cond_cache.pop(idex - 1)
            else:
                condition_latents = vae.encode(conditioning_pixel_values.to(torch.bfloat16))
                condition_latents = torch.stack(condition_latents)
                condition_latents = torch.cat([condition_latents], dim=2)

            B, C, T, H, W = conditioning_pixel_values.shape
            if precompute:
                T = len(cond_images[start:end])
            B, C, lat_t, lat_h, lat_w = condition_latents.shape
            grid_sizes_ref = torch.stack([torch.tensor([lat_t, lat_h // 2, lat_w // 2], dtype=torch.long)])

            if os.environ.get("WAN_MEMLOG") and gpu == 0:
                print(f"[memlog] after_cond_vae: allocated {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
            condition_img =  batch["conditioning_pixel_values"][0, :, 0]
            condition_clip_context = clip.visual([condition_img[:, None, :, :]]).to(dtype=torch.bfloat16, device=device)
            
            mask_len = T
            # With mask_len == T this would re-encode exactly the conditioning frames
            # (same-size bicubic resize is the identity and no zero frames are appended),
            # so reuse the latents computed above instead of a second VAE pass.
            condition_y = condition_latents[0]

            condition_msk_y = get_i2v_mask(lat_t, lat_h, lat_w, mask_len, device=device)
            condition_y = torch.concat([condition_msk_y, condition_y]).to(dtype=torch.bfloat16, device=device)
            
            if use_t5:
                if "context" in const_cache:
                    context, context_ref = const_cache["context"], const_cache["context_ref"]
                else:
                    logger.info('prompt:{}'.format(prompt))
                    context = t5(prompt)[0]
                    context_ref = t5(prompt_ref)[0]
                    if precompute:
                        const_cache["context"], const_cache["context_ref"] = context, context_ref

            max_seq_len = int(math.ceil(np.prod(target_shape) // 4 / cfg.test_cfg.sp_size)) * cfg.test_cfg.sp_size
            target_ref_shape = condition_latents.shape[2:]
            max_seq_len_ref = int(math.ceil(np.prod(target_ref_shape) // 4 / cfg.test_cfg.sp_size)) * cfg.test_cfg.sp_size

            arg_c = {
                    "context":  [context],
                    "seq_len": max_seq_len,
                    "clip_fea": clip_context,
                    "y": [y],
                    "origin_len": origin_len, 
                    "origin_area": origin_area,
                }
            
            arg_ref_c = {
                    "context_ref": [context_ref],
                    "seq_len_ref": max_seq_len_ref,
                    "clip_fea_ref": condition_clip_context,
                    "y_ref": [condition_y]
                }

            if cfg.test_cfg.sample_guide_scale > 1:
                logger.info('sample_guide_scale: {}'.format(cfg.test_cfg.sample_guide_scale))

                if use_t5:
                    context_null = t5(neg_prompt)[0]
                
                arg_null = {
                        "context": [context_null],
                        "seq_len": max_seq_len,
                        "clip_fea": clip_context,
                        "y": [y],
                        "origin_len": origin_len, 
                        "origin_area": origin_area,
                        "is_uncondtion": True
                    }

            if max_seq_len % cfg.test_cfg.sp_size != 0:
                raise ValueError(f"max_seq_len {max_seq_len} is not divisible by sp_size {cfg.test_cfg.sp_size}")

            # cache ref kv
            k_cache = {}
            v_cache = {}
            t = timesteps[0]
            timestep = [t]
            timestep = torch.stack(timestep)
            if os.environ.get("WAN_MEMLOG") and gpu == 0:
                print(f"[memlog] before_forward_ref: allocated {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
            torch.cuda.synchronize(); t_denoise = time.time()
            model(condition_latents, grid_sizes=grid_sizes, k_cache=k_cache, v_cache=v_cache, t=timestep,  method='forward_ref', **arg_ref_c)
            # WAN_CFG_STEPS=n: apply CFG only on the first n steps (expression/structure is
            # set early), the rest run cond-only; unset = CFG on every step
            cfg_steps = int(os.environ.get("WAN_CFG_STEPS", len(timesteps)))
            for i, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                profile = os.environ.get("WAN_PROFILE") == "1" and gpu == 0 and idex == 2 and i == 3
                if profile:
                    prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
                    prof.__enter__()
                noise_pred_cond = TensorList(
                    model(TensorList(latent_model_input), k_cache=k_cache, v_cache=v_cache, t=timestep, grid_sizes_ref=grid_sizes_ref, method='forward_gen', **arg_c)
                )
                if profile:
                    torch.cuda.synchronize()
                    prof.__exit__(None, None, None)
                    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=45), flush=True)

                if cfg.test_cfg.sample_guide_scale > 1 and i < cfg_steps:
                    noise_pred_uncond = TensorList(
                        model(
                            TensorList(latent_model_input), k_cache=k_cache, v_cache=v_cache, t=timestep, grid_sizes_ref=grid_sizes_ref, method='forward_gen', **arg_null
                        )
                    )
                    noise_pred = noise_pred_uncond + cfg.test_cfg.sample_guide_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )
                else:
                    noise_pred = noise_pred_cond

                temp_x0 = sample_scheduler.step(
                    noise_pred[0].unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g,
                )[0]
                latents[0] = temp_x0.squeeze(0)

            torch.cuda.synchronize(); t_decode = time.time()
            # The reference K/V cache (~17 GB/GPU at 1080p) is not needed for decoding
            del k_cache, v_cache
            torch.cuda.empty_cache()
            x0 = latents
            # 将latents转换为float32进行解码
            x0 = [x.to(dtype=torch.float32) for x in x0]
            torch.cuda.empty_cache()
            if split_vae:
                world, rank = dist.get_world_size(), dist.get_rank()
                bounds = _col_bounds(x0[0].shape[-1], world)
                strip = vae.decode_cols(x0[0][:, 1:], bounds[rank], bounds[rank + 1], halo=VAE_HALO)
                out_frames = _gather_cols(strip, [8 * b for b in bounds])[None]
                del strip
            else:
                out_frames = torch.stack(vae.decode([x0[0][:, 1:]]))

            anchor_mode = item.anchor_mode if item.color_anchor else "off"
            if start == 0:
                # The first decoded frame comes out with wrong colors; replace it with
                # its neighbor and take the color/detail reference from the clean frames.
                out_frames[0, :, 0] = out_frames[0, :, 1]
                if anchor_mode == "person":
                    w0 = out_frames[0, :, 1:]
                    color_target = person_stats(w0, person_mask(w0))
                elif anchor_mode == "regions":
                    w0 = out_frames[0, :, 1:]
                    color_target = region_stats(w0, person_mask(w0))
                else:
                    color_target = lab_mean(out_frames[0, :, 1:])
            elif anchor_mode == "person":
                out_frames[0], info = anchor_person(out_frames[0], color_target)
                logger.info(f"[GPU {gpu}] window {idex} person anchor: shift (dL, da, db) "
                            f"{[round(v, 2) for v in info['shift']]}, detail gain {info['detail_gain']:.3f}")
            elif anchor_mode == "regions":
                out_frames[0], info = anchor_regions(out_frames[0], color_target)
                logger.info(f"[GPU {gpu}] window {idex} region anchor: " + ", ".join(
                    f"{k} shift {[round(v, 2) for v in d['shift']]} detail {d['detail_gain']:.3f}" for k, d in info.items()))
            elif anchor_mode == "global":
                out_frames[0], drift = anchor_window_color(out_frames[0], color_target)
                logger.info(f"[GPU {gpu}] window {idex} color drift corrected (dL, da, db): "
                            f"{[round(v, 2) for v in drift.tolist()]}")
            if anchor_mode != "off" and dist.get_world_size() > 1:
                # every rank must condition the next window on identical pixels
                out_frames = out_frames.contiguous()
                dist.broadcast(out_frames, src=0)

            out = (
                rearrange(((out_frames + 1) * 127.5), "1 c t h w -> t h w c")
                .detach()
                .cpu()
                .float()
                .numpy()
                .astype(np.uint8)
            )

            if start != 0:
                out = out[first_num:]

            all_out_frames.extend([*out])
            if gpu == 0:
                torch.cuda.synchronize(); t_end = time.time()
                logger.info(f"[timing] window {idex}: prep/encode {t_denoise - t_window:.1f}s, "
                            f"denoise {t_decode - t_denoise:.1f}s, decode/post {t_end - t_decode:.1f}s")
            start += CLIP_LEN - first_num
            end += CLIP_LEN - first_num
            torch.cuda.empty_cache()
    
    if gpu == 0:
        os.makedirs(os.path.split(output_path)[0], exist_ok=True)
        all_out_frames_array = np.stack(all_out_frames, axis=0)
        all_out_frames_array = all_out_frames_array[:real_frame_len]

        padding_type, padding, side_long = pading_info['padding_type'], pading_info['padding'], pading_info['side_long']
        if padding_type == 'width':
            all_out_frames_array = all_out_frames_array[:, :, padding:padding+side_long, :]
        else:
            all_out_frames_array = all_out_frames_array[:, padding:padding+side_long, :, :]

        video_path = os.path.join(output_path, 'results.mp4')
        write_video(all_out_frames_array, video_path, fps)
        if os.path.exists(audio_save_path):
            merge_video_audio(video_path, audio_save_path)

        res = {'result': video_path, 'status': 'success'}
        return res
    
    return None
        

def worker(gpu, cfg, in_q_list, out_q, initialized_events):
    event = initialized_events[gpu]
    in_q = in_q_list[gpu]
    out_q = out_q

    cfg.gpu = gpu
    cfg.rank = cfg.pmi_rank * cfg.test_cfg.gpu_infer_per_machine + gpu

    init_device = "cpu"
    device = f'cuda:{gpu}'
    vae = None
    model = None
    context = None
    context_null = None

    try:
        torch.cuda.set_device(gpu)
        logger.info(f"[GPU {gpu}] Initializing process group with rank {cfg.rank}, world_size {cfg.world_size}")
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=cfg.rank,
            world_size=cfg.world_size,
            timeout=timedelta(seconds=int(os.getenv("NCCL_INIT_TIMEOUT_SEC", "3600"))),
        )

        if cfg.test_cfg.sp_size > cfg.test_cfg.world_size:
            logger.warning(f"[GPU {gpu}] Reset sp_size to world_size ({cfg.test_cfg.world_size})")
            cfg.test_cfg.sp_size = min(cfg.test_cfg.sp_size, cfg.test_cfg.world_size)

        if cfg.test_cfg.sp_size > 1:
            cfg.model.transformer.use_context_parallel = True
            ops.init_model_parallel_groups(cfg.test_cfg.sp_size)
            assert cfg.test_cfg.sp_size == ops.get_tensor_parallel_world_size()
            logger.info(f"[GPU {gpu}] Tensor parallel world size verified: {ops.get_tensor_parallel_world_size()}")
    
        if gpu == 0:
            logger.info(f"Sharding strategy: {cfg.test_cfg.sharding_strategy}")
            logger.info(f"Param dtype: {cfg.test_cfg.param_dtype}")
            logger.info(f"Reduce dtype: {cfg.test_cfg.reduce_dtype}")
            logger.info(f"Buffer dtype: {cfg.test_cfg.buffer_dtype}")

        # [model] sharding function
        sharding_strategy = get_sharding_strategy(cfg.test_cfg.sharding_strategy.lower())
        
        # Convert dtype strings to torch.dtype
        param_dtype = get_dtype(cfg.test_cfg.param_dtype)
        reduce_dtype = get_dtype(cfg.test_cfg.reduce_dtype)
        buffer_dtype = get_dtype(cfg.test_cfg.buffer_dtype)
        
        shard_fn = partial(
            shard_model,
            device_id=gpu,
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype,
            sharding_strategy=sharding_strategy
        )
        
        t5 = None
        if cfg.model.use_t5:
            with torch.device(init_device):
                t5 = T5Encoder(
                    name=cfg.model.t5_model,
                    text_len=cfg.model.text_len,
                    dtype=torch.bfloat16,
                    device=init_device,
                    checkpoint_path=cfg.model.t5_checkpoint,
                    tokenizer_path=cfg.model.t5_tokenizer,
                )
            t5.model = shard_fn(t5.model, sync_module_states=False)

        try:
            with torch.device(init_device):
                clip = CLIP(
                    name=cfg.model.clip_model,
                    dtype=torch.float16,
                    device=init_device,
                    checkpoint_path=cfg.model.clip_checkpoint,
                    tokenizer_path=cfg.model.clip_tokenizer,
                )
            logger.info(f" --> CLIP: {torch.cuda.memory_allocated()/1e9:.2f} GB, {torch.cuda.memory_reserved()/1e9:.2f} GB")

            
            clip.model.visual = shard_transformer(
                clip.model.visual, 
                {AttentionBlock}, 
                device_id=gpu,
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
                sharding_strategy=sharding_strategy,
            )
            clip.model.textual = clip.model.textual.to(device)
            
            logger.info(f" --> CLIP FSDP: {torch.cuda.memory_allocated()/1e9:.2f} GB, {torch.cuda.memory_reserved()/1e9:.2f} GB")
        except Exception as e:
            logger.error(f"[GPU {gpu}] Failed to initialize VAE: {str(e)}")
            logger.error(f"[GPU {gpu}] Error traceback: {traceback.format_exc()}")
            raise

        try:
            with torch.device(init_device):
                vae = VideoVAE(
                    vae_pth=cfg.model.vae_checkpoint,
                    device=init_device,
                    dtype=torch.bfloat16
                )
            vae.model = vae.model.to(device)
            vae.scale = [vae.scale[0].to(device), vae.scale[1].to(device)]
            logger.info(f" --> VAE: {torch.cuda.memory_allocated()/1e9:.2f} GB, {torch.cuda.memory_reserved()/1e9:.2f} GB")
        except Exception as e:
            logger.error(f"[GPU {gpu}] Failed to initialize VAE: {str(e)}")
            logger.error(f"[GPU {gpu}] Error traceback: {traceback.format_exc()}")
            raise

        # [model] transformer
        try:
            with torch.device('meta'):
                model = build_object_from_dict(cfg.model.transformer)
                torch.set_default_dtype(torch.bfloat16)
        except Exception as e:
            logger.error(f"[GPU {gpu}] Failed to initialize transformer: {str(e)}")
            logger.error(f"[GPU {gpu}] Error traceback: {traceback.format_exc()}")
            raise
        
        cfg.test_cfg.sharding_size = min(cfg.test_cfg.sharding_size, cfg.test_cfg.world_size)
        # prepare fsdp args
        fsdp_mesh = init_device_mesh(
                device_type='cuda', 
                mesh_shape=(dist.get_world_size() // cfg.test_cfg.sharding_size, cfg.test_cfg.sharding_size),
                mesh_dim_names=('replicate', 'shard')
            )
        
        fsdp_kwargs = dict(
                mesh=fsdp_mesh,
                reshard_after_forward=True,
                shard_placement_fn=None,
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.bfloat16,
                    output_dtype=torch.bfloat16,
                    cast_forward_inputs=False
                ),
                offload_policy=None,
                ignored_params=None
            )
        
        for block in model.blocks:
                fully_shard(block, **fsdp_kwargs)
        fully_shard(model, **fsdp_kwargs)
        
        model = model.to_empty(device=device)
        model = model.eval().requires_grad_(False)

        # [model] load checkpoint
        if gpu == 0:
            model_path = cfg.get("model_path", False)
            if model_path:
                sd = load_safetensors(model_path)
                logger.info('gpu: {} model_path: {}'.format(gpu, model_path))
        else:
            sd = {}

        dist.barrier(device_ids=[gpu])
        set_model_state_dict(
            model=model,
            model_state_dict=sd,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
                broadcast_from_rank0=True,
                strict=True
            )
        )
        del sd

        torch.cuda.empty_cache()
        
        # init done
        logger.info(f'[GPU {gpu}] Worker initialization completed')
        if os.environ.get("WAN_MEMLOG") and gpu == 0:
            print(f"[memlog] after init: allocated {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
        event.set()

        # start inference
        logger.info(f"[GPU {gpu}] Starting inference loop")
        while True:
            try:
                # receive data from queue
                item = in_q.get()
                if item.get("overwrite_cfg", None):
                    cfg = merge_configs(cfg, item["overwrite_cfg"])
                res = inference_core(item, gpu, cfg, clip, vae, model, t5, cfg.model.use_t5, context, context_null, device)
                
                if res is not None:
                    out_q.put(res)
                
            except Exception as e:
                logger.error(f"[GPU {gpu}] Error during inference: {str(e)}")
                logger.error(f"[GPU {gpu}] Error traceback: {traceback.format_exc()}")
                if gpu == 0:
                    res = {'result': str(e), 'status': 'fail'}
                    out_q.put(res)
                break

    except Exception as e:
        logger.error(f"[GPU {gpu}] Fatal error: {str(e)}")
        logger.error(f"[GPU {gpu}] Error traceback: {traceback.format_exc()}")
        if gpu == 0:
            res = {'result': str(e), 'status': 'fail'}
            out_q.put(res)

    finally:
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as sync_e:
            logger.warning(f"[GPU {gpu}] cuda.synchronize failed: {sync_e}")

        if dist.is_available() and dist.is_initialized():
            try:
                logger.info(f"[GPU {gpu}] Waiting at barrier before cleanup")
                dist.barrier()
            except Exception as barrier_e:
                logger.warning(f"[GPU {gpu}] barrier failed: {barrier_e}")

            try:
                logger.info(f"[GPU {gpu}] Destroying process group")
                dist.destroy_process_group()
            except Exception as destroy_e:
                logger.warning(f"[GPU {gpu}] destroy_process_group failed: {destroy_e}")
        else:
            logger.info(f"[GPU {gpu}] Process group not initialized, skip barrier/destroy")
        try:
            torch.cuda.empty_cache()
        except Exception as cache_e:
            logger.warning(f"[GPU {gpu}] empty_cache failed: {cache_e}")

        logger.info(f"[GPU {gpu}] Worker cleanup completed")

def main(cfg):
    cfg.pmi_rank = int(os.environ["RANK"])
    cfg.pmi_world_size = int(os.environ["WORLD_SIZE"])
    cfg.test_cfg.gpu_infer_per_machine = torch.cuda.device_count()
    cfg.world_size = cfg.pmi_world_size * cfg.test_cfg.gpu_infer_per_machine
    cfg.test_cfg.world_size = cfg.world_size
    
    in_q_list = [
        torch.multiprocessing.Manager().Queue()
        for _ in range(cfg.test_cfg.gpu_infer_per_machine)
    ]
    out_q = torch.multiprocessing.Manager().Queue()
    initialized_events = [
        torch.multiprocessing.Manager().Event()
        for _ in range(cfg.test_cfg.gpu_infer_per_machine)
    ]

    context = mp.spawn(
        worker,
        nprocs=cfg.test_cfg.gpu_infer_per_machine,
        args=(cfg, in_q_list, out_q, initialized_events),
        join=False,
    )

    all_initialized = False
    while not all_initialized:
        all_initialized = all(event.is_set() for event in initialized_events)
        if not all_initialized:
            time.sleep(0.1)
    logger.info("Inference model is initialized", flush=True)
    inference_pids = context.pids()

    return in_q_list, out_q, inference_pids


class WanAnimate2MPIPipeline(BaseInferencePipeline):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.setup()

    def setup(self):
        if os.getenv("MASTER_ADDR") is None:
            os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDRESS", "localhost")
            os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12345")
            os.environ["RANK"] = os.getenv("rank_id", "0")
            os.environ["WORLD_SIZE"] = os.getenv("WORLD_SIZE", "1")

        self.in_q_list, self.out_q, self.inference_pids = main(self.hparams)    
    
    def __call__(
        self, refer_img_path, tpl_video_path, output_path, width=720, height=1280, fps=24,
        seed=-1, clip_len=81, sample_guide_scale=1.0, step=10,
        prompt=['视频中的人在做动作， 背景静止'],
        prompt_ref=['视频中的人在做动作， 背景静止'],
        color_anchor=True,
        anchor_mode="regions",
        neg_prompt=['过曝，静态，细节模糊不清，字幕，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，三条腿']
    ):
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        logger.info(f"seed: {seed}")

        first_num = 1
        item = EasyDict(
            {
                "height": height,
                "width": width,
                "clip_len": clip_len,
                "refer_len": clip_len,
                "first_num": first_num,
                "refer_img_path": refer_img_path,
                "tpl_video_path": tpl_video_path,
                "fps": fps,
                "seed": seed,
                "output_path": output_path,
                "sample_guide_scale": sample_guide_scale,
                "step": step,
                "prompt": prompt,
                "prompt_ref": prompt_ref,
                "color_anchor": color_anchor,
                "anchor_mode": anchor_mode,
                "neg_prompt": neg_prompt
            }
        )

        logger.info(f"Processing model inference")
        
        for in_q in self.in_q_list:
            in_q.put((item))
        
        result = self.out_q.get()
        return result['result']