"""Upscale a generated video to a target resolution with Real-ESRGAN.

Frames are streamed through ffmpeg (decode -> model -> encode), so memory use does
not grow with video length. The model upscales by its native factor (x2 for
RealESRGAN_x2plus), then frames are resized to the exact target size. Audio from
the input video is copied over unchanged.

Fidelity: Real-ESRGAN is a GAN and can alter content (it deblocks, sharpens and
smooths fine texture). Iterative back-projection then corrects each frame so
that downscaling it reproduces the input frame, which limits the model to adding
detail above the input's resolution; shapes, colors and structure stay those of
the input. Disable with --back-project-iters 0.

Usage:
    python upscale_video.py --input results.mp4 --output results_1080p.mp4 \
        --model RealESRGAN_x2plus.pth --short-side 1080
"""

import argparse
import json
import subprocess

import numpy as np
import torch
import torch.nn.functional as F


def probe_video(path):
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate", "-of", "json", path,
        ],
        check=True, capture_output=True, text=True,
    ).stdout
    stream = json.loads(out)["streams"][0]
    return stream["width"], stream["height"], stream["r_frame_rate"]


def target_size(width, height, short_side):
    scale = short_side / min(width, height)
    # libx264 + yuv420p requires even dimensions
    return round(width * scale / 2) * 2, round(height * scale / 2) * 2


def load_model(model_path, device):
    import spandrel

    descriptor = spandrel.ModelLoader().load_from_file(model_path)
    descriptor = descriptor.to(device).eval()
    if descriptor.supports_half:
        descriptor = descriptor.half()
    return descriptor


def back_project(up, src, iters):
    """Enforce area_downsample(up) == src via iterative back-projection."""
    for _ in range(iters):
        residual = src - F.interpolate(up, size=src.shape[-2:], mode="area")
        up = up + F.interpolate(residual, size=up.shape[-2:], mode="bicubic")
    return up


@torch.no_grad()
def upscale_video(input_path, output_path, model_path, short_side=1080, batch_size=4,
                  crf=16, back_project_iters=5, device="cuda"):
    width, height, fps = probe_video(input_path)
    out_w, out_h = target_size(width, height, short_side)
    print(f"Upscaling {width}x{height} -> {out_w}x{out_h} @ {fps} fps")

    model = load_model(model_path, device)
    dtype = torch.float16 if model.supports_half else torch.float32

    decoder = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", input_path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE,
    )
    encoder = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{out_w}x{out_h}", "-r", fps, "-i", "-",
            "-i", input_path,
            "-map", "0:v", "-map", "1:a?", "-c:a", "copy",
            "-c:v", "libx264", "-crf", str(crf), "-preset", "slow", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", output_path,
        ],
        stdin=subprocess.PIPE,
    )

    frame_bytes = width * height * 3
    num_frames = 0

    def flush(batch):
        x = torch.from_numpy(np.stack(batch)).to(device)
        src = x.permute(0, 3, 1, 2).float() / 255.0
        x = model(src.to(dtype)).float() if out_h > height else src
        x = F.interpolate(x, size=(out_h, out_w), mode="bicubic", antialias=True)
        x = back_project(x, src, back_project_iters)
        x = (x.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        encoder.stdin.write(x.tobytes())

    try:
        batch = []
        while True:
            buf = decoder.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            batch.append(np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3))
            if len(batch) == batch_size:
                flush(batch)
                num_frames += len(batch)
                batch = []
        if batch:
            flush(batch)
            num_frames += len(batch)
    finally:
        decoder.stdout.close()
        encoder.stdin.close()
        decoder.wait()
        encoder.wait()

    if decoder.returncode != 0 or encoder.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (decoder={decoder.returncode}, encoder={encoder.returncode})"
        )
    print(f"Wrote {num_frames} frames to {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True, help="Real-ESRGAN (or other spandrel-supported) weights")
    parser.add_argument("--short-side", type=int, default=1080)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crf", type=int, default=16)
    parser.add_argument("--back-project-iters", type=int, default=5,
                        help="consistency iterations with the input (0 = raw model output)")
    args = parser.parse_args()
    upscale_video(args.input, args.output, args.model, args.short_side, args.batch_size, args.crf,
                  args.back_project_iters)


def _read_frames(path, width, height):
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE,
    )
    n = width * height * 3
    try:
        while True:
            buf = proc.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
    finally:
        proc.stdout.close()
        proc.wait()


@torch.no_grad()
def back_project_video(src_path, up_path, output_path, iters=5, crf=16, device="cuda"):
    """Apply back-projection to an already-upscaled video (e.g. from SeedVR2), using
    `src_path` as the low-resolution reference; audio is taken from `src_path`."""
    sw, sh, fps = probe_video(src_path)
    uw, uh, _ = probe_video(up_path)
    encoder = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{uw}x{uh}", "-r", fps, "-i", "-",
            "-i", src_path, "-map", "0:v", "-map", "1:a?", "-c:a", "copy",
            "-c:v", "libx264", "-crf", str(crf), "-preset", "slow", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", output_path,
        ],
        stdin=subprocess.PIPE,
    )
    for src, up in zip(_read_frames(src_path, sw, sh), _read_frames(up_path, uw, uh)):
        s = torch.from_numpy(src.copy()).to(device).permute(2, 0, 1)[None].float() / 255.0
        u = torch.from_numpy(up.copy()).to(device).permute(2, 0, 1)[None].float() / 255.0
        u = back_project(u, s, iters)
        encoder.stdin.write((u.clamp(0, 1) * 255).round().to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy().tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    return output_path
