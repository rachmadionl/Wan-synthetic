"""Run Wan-Animate-2 inference on Modal: generate at 720P, then upscale to 1080p.

Usage:
    # 1. One-time: download checkpoints (~82 GB) into a Modal Volume
    modal run modal_app.py::download_weights

    # 2. Generate (Distillation, 10 steps, no CFG -- fastest) and upscale to 1080p
    modal run modal_app.py \
        --refer-img ./examples/demo1/reference.png \
        --refer-video ./examples/demo1/template.mp4 \
        --prompt "$(cat examples/demo1/prompt.txt)"

    # Base model (40 steps, CFG 3.0)
    modal run modal_app.py --variant base ...

    # Skip upscaling and keep the native 720P output
    modal run modal_app.py --short-side 0 ...

Resolution:
    Inputs of any resolution (e.g. 1080p) are resized by the pipeline to the area
    of --width x --height, keeping their aspect ratio. Generation runs at 720P
    (the highest resolution the shipped configs target); a Real-ESRGAN x2 pass
    then brings the result to --short-side (default 1080).

GPU selection:
    The 14B transformer + T5 + CLIP + VAE weigh ~50 GB, and at 720P the cached
    reference-video K/V adds ~62 GB; both are sharded across GPUs. The default
    4x H100 leaves comfortable headroom at 720P. 2x H100 may fit but is tight;
    2x A100-80GB is the minimum for 480P per the upstream README. The GPU count
    must divide the 40 attention heads (1, 2, 4, 5, 8). Override with e.g.:
        WAN_GPU=H100:8 modal run modal_app.py ...
"""

import os
from pathlib import Path

# The workspace default (2023.12) installs Modal's runtime deps into the same env as
# requirements.txt, where its pins (e.g. aiohttp 3.8) clash with ours. 2024.10+ isolates them.
os.environ.setdefault("MODAL_IMAGE_BUILDER_VERSION", "2025.06")

import modal

APP_NAME = "wan-animate-2"
LOCAL_DIR = Path(__file__).parent
REPO_DIR = "/root/Wan-Animate-2"
CKPT_DIR = "/ckpts"
HF_REPO = "Wan-AI/Wan2.2-Animate-2-14B"

GPU = os.environ.get("WAN_GPU", "H100:4")
N_GPUS = int(GPU.split(":")[1]) if ":" in GPU else 1
UPSCALE_GPU = os.environ.get("WAN_UPSCALE_GPU", "L40S")

CONFIGS = {
    "base": "wan_animate_2.yaml",
    "distillation": "wan_animate_2_distillation.yaml",
}

FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
    "flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
)
ESRGAN_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
ESRGAN_PATH = "/root/weights/RealESRGAN_x2plus.pth"

ckpt_volume = modal.Volume.from_name("wan-animate-2-ckpts", create_if_missing=True)
# Persists torch.compile / triton caches (flex_attention is compiled with max-autotune)
cache_volume = modal.Volume.from_name("wan-animate-2-cache", create_if_missing=True)

# Only code goes into the image: data folders, videos and outputs placed in the repo
# directory must neither bloat it nor break builds while they are being copied in.
_CODE_ROOTS = {"core", "pipelines", "wanxiang", "infer", "requirements.txt", "setup.py"}


def _not_code(path: Path) -> bool:
    parts = Path(path).parts
    if not parts or parts[0] not in _CODE_ROOTS:
        return True
    return "__pycache__" in parts or Path(path).suffix not in {"", ".py", ".yaml", ".txt"}


image = (
    # CUDA devel base: nvcc is needed to build SageAttention. Ubuntu 22.04 has glibc 2.35
    # (the prebuilt flash-attn wheel needs >= 2.32).
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11")
    # ffmpeg: audio extraction/merging; gcc: torch.compile (flex_attention) + triton
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0", "build-essential", "git", "wget")
    # requirements.txt pins torch==2.7.1+cu126, which only exists on the PyTorch index
    .pip_install_from_requirements(
        str(LOCAL_DIR / "requirements.txt"), extra_index_url="https://download.pytorch.org/whl/cu126"
    )
    .pip_install(FLASH_ATTN_WHEEL, "huggingface_hub[hf_transfer]", "spandrel==0.4.2")
    # FlashAttention-3 (Hopper), used with WAN_ATTN=fa3. Only the kernels this model needs
    # (forward, bf16 + fp8, head dim 128, no varlen/paging/softcap/...) to keep the compile short.
    .pip_install("ninja", "packaging", "wheel", "setuptools")
    .run_commands(
        "git clone --depth 1 --branch v2.8.3 https://github.com/Dao-AILab/flash-attention.git /opt/flash-attention"
        " && cd /opt/flash-attention/hopper"
        " && export CC=gcc CXX=g++ LDSHARED='g++ -shared' MAX_JOBS=16 NVCC_THREADS=2"
        + "".join(f" FLASH_ATTENTION_DISABLE_{f}=TRUE" for f in (
            "BACKWARD", "SPLIT", "PAGEDKV", "APPENDKV", "LOCAL", "SOFTCAP", "PACKGQA", "FP16",
            "HDIM64", "HDIM96", "HDIM192", "HDIM256", "HDIMDIFF64", "HDIMDIFF192", "SM80", "VARLEN", "CLUSTER"))
        + " && python -m pip install . --no-build-isolation"
    )
    .run_commands(f"mkdir -p {Path(ESRGAN_PATH).parent} && wget -q -O {ESRGAN_PATH} {ESRGAN_URL}")
    .env(
        {
            "PYTHONPATH": REPO_DIR,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TORCHINDUCTOR_CACHE_DIR": "/root/.cache/inductor",
            # /root/.cache is a volume mount; keep baked-in torchvision weights elsewhere
            "TORCH_HOME": "/opt/torch_home",
        }
    )
    # person segmentation used by anchor_mode="person"
    .run_commands("python -c \"from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large as m, "
                  "DeepLabV3_MobileNet_V3_Large_Weights as W; m(weights=W.DEFAULT)\"")
    .add_local_dir(
        LOCAL_DIR,
        REPO_DIR,
        copy=True,  # the pipeline writes ./logs and outputs relative to the repo
        ignore=_not_code,
    )
)

SEEDVR2_DIR = "/opt/seedvr2"
SEEDVR2_MODELS = "/opt/seedvr2/models/SEEDVR2"
SEEDVR2_DIT = os.environ.get("WAN_SEEDVR2_DIT", "seedvr2_ema_3b_fp16.safetensors")

# Separate image for the SeedVR2 diffusion upscaler (numz/ComfyUI-SeedVR2_VideoUpscaler CLI)
seedvr2_image = (
    modal.Image.from_registry("python:3.12-slim-bookworm")
    .apt_install("ffmpeg", "git", "build-essential", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.8.0", "torchvision==0.23.0", index_url="https://download.pytorch.org/whl/cu128")
    .run_commands(
        f"git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git {SEEDVR2_DIR}"
        f" && cd {SEEDVR2_DIR} && git checkout v2.5.23",
        f"pip install -r {SEEDVR2_DIR}/requirements.txt huggingface_hub[hf_transfer]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_commands(
        "python -c \"from huggingface_hub import hf_hub_download as d; "
        f"[d('numz/SeedVR2_comfyUI', f, local_dir='{SEEDVR2_MODELS}') for f in "
        "['seedvr2_ema_3b_fp16.safetensors', 'ema_vae_fp16.safetensors']]\""
    )
    .add_local_file(LOCAL_DIR / "infer/upscale_video.py", "/root/upscale_video.py", copy=True)
)

SEEDVR2_DIR = "/opt/seedvr2"
SEEDVR2_MODELS = "/opt/seedvr2/models/SEEDVR2"
SEEDVR2_DIT = os.environ.get("WAN_SEEDVR2_DIT", "seedvr2_ema_3b_fp16.safetensors")

# Separate image for the SeedVR2 diffusion upscaler (numz/ComfyUI-SeedVR2_VideoUpscaler CLI)
seedvr2_image = (
    modal.Image.from_registry("python:3.12-slim-bookworm")
    .apt_install("ffmpeg", "git", "build-essential", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.8.0", "torchvision==0.23.0", index_url="https://download.pytorch.org/whl/cu128")
    .run_commands(
        f"git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git {SEEDVR2_DIR}"
        f" && cd {SEEDVR2_DIR} && git checkout v2.5.23",
        f"pip install -r {SEEDVR2_DIR}/requirements.txt huggingface_hub[hf_transfer]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_commands(
        "python -c \"from huggingface_hub import hf_hub_download as d; "
        f"[d('numz/SeedVR2_comfyUI', f, local_dir='{SEEDVR2_MODELS}') for f in "
        "['seedvr2_ema_3b_fp16.safetensors', 'ema_vae_fp16.safetensors']]\""
    )
    .add_local_file(LOCAL_DIR / "infer/upscale_video.py", "/root/upscale_video.py", copy=True)
)

app = modal.App(APP_NAME, image=image)


@app.function(
    volumes={CKPT_DIR: ckpt_volume},
    timeout=3 * 60 * 60,
    cpu=8,
    memory=16 * 1024,
)
def download_weights(force: bool = False):
    from huggingface_hub import snapshot_download

    marker = Path(CKPT_DIR) / ".download_complete"
    if marker.exists() and not force:
        print("Checkpoints already present; pass --force to re-download.")
        return

    snapshot_download(repo_id=HF_REPO, local_dir=CKPT_DIR, ignore_patterns=["assets/*"])
    marker.touch()
    ckpt_volume.commit()
    print("Downloaded checkpoints:")
    for p in sorted(Path(CKPT_DIR).rglob("*")):
        if p.is_file():
            print(f"  {p.stat().st_size / 1e9:8.2f} GB  {p.relative_to(CKPT_DIR)}")


@app.cls(
    gpu=GPU,
    # pass-through switches for A/B tests and profiling (read when the app is launched)
    env={k: os.environ[k] for k in ("WAN_ATTN", "WAN_PROFILE", "WAN_LOG_SCALE", "WAN_MEMLOG", "WAN_PRECOMPUTE", "WAN_SPLIT_VAE", "WAN_CFG_STEPS") if k in os.environ},
    volumes={CKPT_DIR: ckpt_volume, "/root/.cache": cache_volume},
    # rank 0 materializes the full 33 GB state dict in host RAM, and every rank
    # loads T5 (11 GB) on CPU before sharding.
    memory=(64 + 24 * N_GPUS) * 1024,
    cpu=2 * N_GPUS,
    # ~28 s of 4x H100 time per second of output video at 720P, so a 60 s input
    # takes ~30 min; leave room for longer inputs.
    timeout=2 * 60 * 60,
    # @modal.enter loads ~45 GB of weights from the volume and spawns one worker per GPU
    startup_timeout=30 * 60,
    # Idle 4x H100 costs ~$18/hr; keep warm only briefly for back-to-back requests
    scaledown_window=2 * 60,
)
class WanAnimate2:
    variant: str = modal.parameter(default="distillation")

    @modal.enter()
    def load(self):
        import torch

        if self.variant not in CONFIGS:
            raise ValueError(f"variant must be one of {list(CONFIGS)}, got {self.variant!r}")
        if not (Path(CKPT_DIR) / ".download_complete").exists():
            raise RuntimeError("Checkpoints missing. Run: modal run modal_app.py::download_weights")

        print(f"Visible GPUs: {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}")
        # The YAML configs reference checkpoints as ../ckpts/... (relative to infer/);
        # point them at the volume mount instead.
        os.chdir(f"{REPO_DIR}/infer")
        config = Path(CONFIGS[self.variant]).read_text().replace("../ckpts/", f"{CKPT_DIR}/")
        if os.environ.get("WAN_LOG_SCALE"):
            # experiment knob: logit bias on the frame carried over from the previous window
            import re
            config = re.sub(r"log_scale: *[-0-9.]+", f"log_scale: {float(os.environ['WAN_LOG_SCALE'])}", config)
            print(f"log_scale override: {os.environ['WAN_LOG_SCALE']}")
        config_path = f"/tmp/{CONFIGS[self.variant]}"
        Path(config_path).write_text(config)

        from core import build_object_from_config_file

        self.pipeline = build_object_from_config_file(config_path)

    @modal.method()
    def generate(
        self,
        refer_img: bytes,
        refer_video: bytes,
        prompt: str,
        width: int = 720,
        height: int = 1280,
        fps: int = 25,
        clip_len: int = 81,
        step: int = 10,
        guide_scale: float = 1.0,
        seed: int = -1,
        prompt_ref: str = "人物动作的参考视频",
        neg_prompt: str = "",
        color_anchor: bool = True,
        anchor_mode: str = "regions",
    ) -> bytes:
        import tempfile
        import time

        import torch

        session_dir = tempfile.mkdtemp(prefix="session_", dir="/tmp")
        img_path = os.path.join(session_dir, "reference.png")
        video_path = os.path.join(session_dir, "template.mp4")
        Path(img_path).write_bytes(refer_img)
        Path(video_path).write_bytes(refer_video)

        tic = time.time()
        result = self.pipeline(
            refer_img_path=img_path,
            tpl_video_path=video_path,
            output_path=session_dir,
            width=width,
            height=height,
            fps=fps,
            seed=seed,
            clip_len=clip_len,
            sample_guide_scale=guide_scale,
            step=step,
            prompt=prompt,
            prompt_ref=prompt_ref,
            **({"neg_prompt": [neg_prompt]} if neg_prompt else {}),
            color_anchor=color_anchor,
            anchor_mode=anchor_mode,
        )
        if not (isinstance(result, str) and os.path.isfile(result)):
            raise RuntimeError(f"Inference failed: {result}")
        print(f"Processing time: {time.time() - tic:.1f}s")
        # Inference runs in worker subprocesses, so query device memory directly
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            print(f"GPU {i}: {(total - free) / 1e9:.1f} / {total / 1e9:.1f} GB in use")
        return Path(result).read_bytes()


@app.function(gpu=UPSCALE_GPU, timeout=2 * 60 * 60, scaledown_window=60, cpu=8, memory=32 * 1024)
def upscale(video: bytes, short_side: int = 1080) -> bytes:
    import tempfile

    from infer.upscale_video import upscale_video

    work_dir = tempfile.mkdtemp(dir="/tmp")
    in_path, out_path = f"{work_dir}/in.mp4", f"{work_dir}/out.mp4"
    Path(in_path).write_bytes(video)
    upscale_video(in_path, out_path, ESRGAN_PATH, short_side=short_side)
    return Path(out_path).read_bytes()


@app.local_entrypoint()
def main(
    refer_img: str = str(LOCAL_DIR / "examples/demo1/reference.png"),
    refer_video: str = str(LOCAL_DIR / "examples/demo1/template.mp4"),
    prompt: str = "",
    variant: str = "distillation",
    width: int = 720,
    height: int = 1280,
    fps: int = 25,
    clip_len: int = 81,
    step: int = 0,
    guide_scale: float = -1.0,
    seed: int = -1,
    short_side: int = 1080,
    output: str = "outputs/modal_result.mp4",
):
    if not prompt:
        prompt = (LOCAL_DIR / "examples/demo1/prompt.txt").read_text().strip()
    # Per-variant defaults from the README
    if step <= 0:
        step = 10 if variant == "distillation" else 40
    if guide_scale < 0:
        guide_scale = 1.0 if variant == "distillation" else 3.0

    video = WanAnimate2(variant=variant).generate.remote(
        refer_img=Path(refer_img).read_bytes(),
        refer_video=Path(refer_video).read_bytes(),
        prompt=prompt,
        width=width,
        height=height,
        fps=fps,
        clip_len=clip_len,
        step=step,
        guide_scale=guide_scale,
        seed=seed,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(video)
    print(f"Saved {output}")

    if short_side > 0:
        upscaled = output.with_name(f"{output.stem}_{short_side}p{output.suffix}")
        upscaled.write_bytes(upscale.remote(video, short_side=short_side))
        print(f"Saved {upscaled}")


@app.local_entrypoint()
def batch(
    frames_dir: str,
    driving: str,
    out_dir: str,
    captions: str = "",
    only: str = "",
    variant: str = "distillation",
    width: int = 720,
    height: int = 1280,
    fps: int = 25,
    seed: int = 42,
    short_side: int = 1080,
    max_containers: int = 2,
    color_anchor: bool = True,
    clip_len: int = 81,
    anchor_mode: str = "regions",
    final_size: str = "",
    steps: int = 0,
    prompt_ref: str = "人物动作的参考视频",
    guide_scale: float = -1.0,
    neg_prompt: str = "",
):
    """Animate every reference image in --frames-dir with one driving video.

    Prompts come from --captions (JSON mapping image stem -> caption; defaults to
    <frames-dir>/captions.json). Outputs that already exist are skipped, so the
    command can be re-run as more inputs arrive. --only takes comma-separated
    stems to run a subset.

        modal run modal_app.py::batch --frames-dir first_frames \\
            --driving motion_25fps.mp4 --out-dir outputs
    """
    import json

    frames = sorted(p for p in Path(frames_dir).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if only:
        wanted = {s.strip() for s in only.split(",")}
        frames = [p for p in frames if p.stem in wanted]
    prompts = json.loads(Path(captions or Path(frames_dir) / "captions.json").read_text())
    missing = [p.stem for p in frames if not prompts.get(p.stem)]
    if missing:
        raise SystemExit(f"No caption for: {missing}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # --final-size WxH: resize the generated video locally (e.g. native 1904x1072 -> 1920x1080);
    # the raw generation is kept in <out-dir>/intermediate/
    if final_size:
        final_suffix = f"_{final_size.split('x')[1]}p"
        raw_dir = out / "intermediate"
        raw_dir.mkdir(exist_ok=True)
    else:
        final_suffix = f"_{short_side}p" if short_side > 0 else ""
        raw_dir = out
    todo = [p for p in frames if not (out / f"{p.stem}{final_suffix}.mp4").exists()]
    print(f"{len(frames) - len(todo)} done already, {len(todo)} to run")
    if not todo:
        return

    driving_bytes = Path(driving).read_bytes()
    step = steps or (10 if variant == "distillation" else 40)
    if guide_scale < 0:
        guide_scale = 1.0 if variant == "distillation" else 3.0
    model = WanAnimate2.with_options(max_containers=max_containers)(variant=variant)

    calls = {
        p.stem: model.generate.spawn(
            refer_img=p.read_bytes(), refer_video=driving_bytes, prompt=prompts[p.stem],
            width=width, height=height, fps=fps, step=step, guide_scale=guide_scale, seed=seed,
            color_anchor=color_anchor, clip_len=clip_len, anchor_mode=anchor_mode,
            prompt_ref=prompt_ref, neg_prompt=neg_prompt,
        )
        for p in todo
    }
    failed = []
    upscales = {}
    for name, call in calls.items():
        try:
            video = call.get()
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            failed.append(name)
            continue
        (raw_dir / f"{name}.mp4").write_bytes(video)
        if final_size:
            import subprocess

            w, h = final_size.split("x")
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-i", str(raw_dir / f"{name}.mp4"),
                 # The raw generation is untagged BT.601 (ffmpeg's rgb24->yuv default); players
                 # assume BT.709 for HD, which shifts skin tones. Convert to BT.709 and tag it.
                 "-vf", f"scale={w}:{h}:flags=lanczos:in_color_matrix=bt601:in_range=tv:"
                        "out_color_matrix=bt709:out_range=tv,"
                        "setparams=range=tv:color_primaries=bt709:color_trc=bt709:colorspace=bt709",
                 "-c:v", "libx264", "-crf", "14",
                 "-preset", "slow", "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart",
                 str(out / f"{name}{final_suffix}.mp4")],
                check=True,
            )
            print(f"[final] {name}{final_suffix}.mp4")
            continue
        print(f"[gen ] {name}.mp4")
        if short_side > 0:
            upscales[name] = upscale.spawn(video, short_side=short_side)
    for name, call in upscales.items():
        try:
            (out / f"{name}{final_suffix}.mp4").write_bytes(call.get())
            print(f"[1080] {name}{final_suffix}.mp4")
        except Exception as e:
            print(f"[FAIL] upscale {name}: {e}")
            failed.append(name)
    print(f"Finished: {len(todo) - len(failed)} ok, {len(failed)} failed {failed or ''}")


@app.function(
    image=seedvr2_image,
    gpu=os.environ.get("WAN_SEEDVR2_GPU", "H100"),
    timeout=2 * 60 * 60,
    cpu=8,
    memory=64 * 1024,
    scaledown_window=60,
)
def upscale_seedvr2(video: bytes, short_side: int = 1080, cli_args: list[str] = (),
                    back_project_iters: int = 0) -> bytes:
    """Upscale with SeedVR2 (one-step diffusion VSR). With cli_args=["--help"], returns the CLI help."""
    import subprocess
    import sys
    import tempfile
    import time

    if list(cli_args) == ["--help"]:
        r = subprocess.run([sys.executable, "inference_cli.py", "--help"], cwd=SEEDVR2_DIR,
                           capture_output=True, text=True)
        return (r.stdout + r.stderr).encode()

    work = tempfile.mkdtemp(dir="/tmp")
    src, raw, out = f"{work}/in.mp4", f"{work}/seedvr2.mp4", f"{work}/out.mp4"
    Path(src).write_bytes(video)
    cmd = [
        sys.executable, "inference_cli.py", src, "--output", raw,
        "--model_dir", SEEDVR2_MODELS, "--dit_model", SEEDVR2_DIT,
        "--resolution", str(short_side), *cli_args,
    ]
    print(" ".join(cmd), flush=True)
    tic = time.time()
    subprocess.run(cmd, cwd=SEEDVR2_DIR, check=True)
    print(f"SeedVR2 time: {time.time() - tic:.1f}s", flush=True)

    sys.path.insert(0, "/root")
    from upscale_video import back_project_video, probe_video

    # The CLI (v2.5.23) leaves --prepend_frames in its output; drop the leading extras
    # so frames stay aligned with the input (and its audio).
    def count(p):
        return int(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                                   "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", p],
                                  capture_output=True, text=True, check=True).stdout.strip())
    extra = count(raw) - count(src)
    if extra > 0:
        fps = probe_video(src)[2]
        trimmed = f"{work}/seedvr2_trim.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", raw, "-vf", f"select=gte(n\\,{extra}),setpts=N/({fps})/TB",
                        "-r", fps, "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", trimmed], check=True)
        raw = trimmed
        print(f"Dropped {extra} leading frames", flush=True)
    print("SeedVR2 output:", probe_video(raw), flush=True)
    if back_project_iters > 0:
        back_project_video(src, raw, out, iters=back_project_iters)
    else:
        # restore audio from the input
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", raw, "-i", src, "-map", "0:v", "-map", "1:a?",
                        "-c", "copy", out], check=True)
    return Path(out).read_bytes()


@app.local_entrypoint()
def seedvr2(input: str = "", output: str = "", short_side: int = 1080, args: str = "",
            back_project_iters: int = 0, show_help: bool = False):
    """modal run modal_app.py::seedvr2 --input in.mp4 --output out.mp4 [--args "--batch_size 33 ..."]"""
    import shlex

    if show_help:
        print(upscale_seedvr2.remote(b"", cli_args=["--help"]).decode())
        return
    data = upscale_seedvr2.remote(Path(input).read_bytes(), short_side=short_side,
                                  cli_args=shlex.split(args), back_project_iters=back_project_iters)
    Path(output).write_bytes(data)
    print(f"Saved {output}")
