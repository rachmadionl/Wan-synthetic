import torch
import torch.nn.functional as F
import kornia


def lab_mean(frames, scale=0.25):
    """Mean CIE Lab color of a clip. frames: (3, T, H, W) in [-1, 1]."""
    x = frames.permute(1, 0, 2, 3).float()
    x = F.interpolate(x, scale_factor=scale, mode="area")
    lab = kornia.color.rgb_to_lab(((x + 1) / 2).clamp(0, 1))
    return lab.mean(dim=(0, 2, 3))


@torch.no_grad()
def shift_lab(frames, delta, chunk=16):
    """Add a per-frame Lab offset. frames: (3, T, H, W) in [-1, 1]; delta: (T, 3)."""
    out = torch.empty_like(frames)
    for s in range(0, frames.shape[1], chunk):
        x = frames[:, s:s + chunk].permute(1, 0, 2, 3).float()
        lab = kornia.color.rgb_to_lab(((x + 1) / 2).clamp(0, 1))
        lab = lab + delta[s:s + chunk, :, None, None].to(lab)
        rgb = kornia.color.lab_to_rgb(lab).clamp(0, 1) * 2 - 1
        out[:, s:s + chunk] = rgb.permute(1, 0, 2, 3).to(frames.dtype)
    return out


def anchor_window_color(frames, target, tail=16, max_shift=(8.0, 4.0, 4.0)):
    """Pull a generated window's color back to `target` (a Lab mean).

    Each window is conditioned on the last frame of the previous one, so color
    error compounds over a long video. The window's first frame already matches
    the (corrected) conditioning frame, and drift accumulates roughly linearly
    from there, so the correction is ramped from 0 at the first frame to the full
    offset, measured on the last `tail` frames, at the end. The offset is clamped
    so pose/composition changes between windows cannot cause large shifts.
    """
    drift = target.to(frames.device) - lab_mean(frames[:, -tail:])
    limit = torch.tensor(max_shift, device=drift.device)
    drift = torch.maximum(torch.minimum(drift, limit), -limit)
    ramp = torch.linspace(0, 1, frames.shape[1], device=drift.device)
    return shift_lab(frames, ramp[:, None] * drift[None, :]), drift


# ---------------------------------------------------------------------------
# Person-aware anchoring: color and fine-detail level inside a person mask.
# Over long videos the window-to-window chain makes the person progressively
# crisper / higher-contrast (detail energy grows ~20-40% over 30 s) while the
# background stays stable, so whole-frame statistics cannot see it.
# ---------------------------------------------------------------------------
_SEG = None


@torch.no_grad()
def person_mask(frames, size=320, chunk=16):
    """Soft person mask (T, 1, H, W) in [0, 1] (DeepLabV3-MobileNetV3, VOC class 15).
    frames: (3, T, H, W) in [-1, 1]."""
    global _SEG
    from torchvision.models.segmentation import (
        deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights)
    if _SEG is None:
        _SEG = deeplabv3_mobilenet_v3_large(
            weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT).eval().to(frames.device)
    T, H, W = frames.shape[1:]
    mean = torch.tensor([0.485, 0.456, 0.406], device=frames.device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=frames.device)[:, None, None]
    out = []
    for s in range(0, T, chunk):
        x = frames[:, s:s + chunk].permute(1, 0, 2, 3).float().add(1).div(2)
        x = F.interpolate(x, size=(size, int(round(size * W / H))), mode="bilinear", antialias=True)
        out.append(_SEG((x - mean) / std)["out"].softmax(1)[:, 15:16])
    m = kornia.filters.gaussian_blur2d(torch.cat(out), (9, 9), (3.0, 3.0))
    return F.interpolate(m, size=(H, W), mode="bilinear").clamp(0, 1)


def _detail_sigma(height):
    return 2.0 * height / 720


@torch.no_grad()
def person_stats(frames, mask, chunk=16):
    """Lab mean (3,) and mean |high-pass| detail energy inside the mask."""
    sigma = _detail_sigma(frames.shape[2])
    k = int(2 * round(3 * sigma) + 1)
    lab_sum, e_sum, w_sum = 0, 0, 0
    for s in range(0, frames.shape[1], chunk):
        x = frames[:, s:s + chunk].permute(1, 0, 2, 3).float()
        m = mask[s:s + chunk]
        lab = kornia.color.rgb_to_lab(((x + 1) / 2).clamp(0, 1))
        hp = (x - kornia.filters.gaussian_blur2d(x, (k, k), (sigma, sigma))).abs().mean(1, keepdim=True)
        lab_sum = lab_sum + (lab * m).sum(dim=(0, 2, 3))
        e_sum = e_sum + (hp * m).sum()
        w_sum = w_sum + m.sum()
    w_sum = w_sum.clamp_min(1e-6)
    return lab_sum / w_sum, e_sum / w_sum


@torch.no_grad()
def anchor_person(frames, target, tail=16, max_shift=(6.0, 3.0, 3.0), min_detail=0.75, chunk=16):
    """Pull the person's color and detail level in a window back to `target`.

    target = (lab_mean, detail_energy) measured on the first window. Detail is only
    ever damped (never sharpened), down to `min_detail`, by blending the high-pass
    band; both corrections ramp from 0 at the window's first frame (which matches the
    already-corrected conditioning frame) to full at the end, and are applied
    through the soft person mask only.
    """
    mask = person_mask(frames)
    t_lab, t_energy = (t.to(frames.device) for t in target)
    lab_now, e_now = person_stats(frames[:, -tail:], mask[-tail:])
    lim = torch.tensor(max_shift, device=frames.device)
    shift = torch.maximum(torch.minimum(t_lab - lab_now, lim), -lim)
    alpha = (t_energy / e_now.clamp_min(1e-6)).clamp(min_detail, 1.0)

    sigma = _detail_sigma(frames.shape[2])
    k = int(2 * round(3 * sigma) + 1)
    T = frames.shape[1]
    ramp = torch.linspace(0, 1, T, device=frames.device)
    out = torch.empty_like(frames)
    for s in range(0, T, chunk):
        x = frames[:, s:s + chunk].permute(1, 0, 2, 3).float()
        r = ramp[s:s + chunk, None, None, None]
        m = mask[s:s + chunk]
        low = kornia.filters.gaussian_blur2d(x, (k, k), (sigma, sigma))
        a = 1 - r * (1 - alpha)                       # detail gain, 1 -> alpha over the window
        y = low + a * (x - low)
        lab = kornia.color.rgb_to_lab(((y + 1) / 2).clamp(0, 1)) + r * shift[None, :, None, None]
        y = kornia.color.lab_to_rgb(lab).clamp(0, 1) * 2 - 1
        out[:, s:s + chunk] = (m * y + (1 - m) * x).permute(1, 0, 2, 3).to(frames.dtype)
    return out, {"shift": shift.tolist(), "detail_gain": alpha.item()}


@torch.no_grad()
def region_stats(frames, mask):
    """Targets for anchor_regions: (person, background) stats, each (lab_mean, detail_energy)."""
    return person_stats(frames, mask), person_stats(frames, 1 - mask)


@torch.no_grad()
def anchor_regions(frames, targets, tail=16, max_shift=(6.0, 3.0, 3.0), min_detail=(0.75, 0.5), chunk=16):
    """anchor_person applied separately to the person and to the background.

    The background drifts too (its fine detail can grow ~2.7x over 30 s and it
    brightens), so each region is pulled back to its own window-1 color and detail
    level, and the two results are composited through the soft person mask.
    min_detail is (person, background): the background barely changes with pose,
    so it can be damped further per window without risking false corrections.
    """
    mask = person_mask(frames)
    sigma = _detail_sigma(frames.shape[2])
    k = int(2 * round(3 * sigma) + 1)
    lim = torch.tensor(max_shift, device=frames.device)
    corr = []
    for (t_lab, t_energy), m, floor in zip(targets, (mask, 1 - mask), min_detail):
        lab_now, e_now = person_stats(frames[:, -tail:], m[-tail:])
        shift = torch.maximum(torch.minimum(t_lab.to(frames.device) - lab_now, lim), -lim)
        alpha = (t_energy.to(frames.device) / e_now.clamp_min(1e-6)).clamp(floor, 1.0)
        corr.append((shift, alpha))

    T = frames.shape[1]
    ramp = torch.linspace(0, 1, T, device=frames.device)
    out = torch.empty_like(frames)
    for s in range(0, T, chunk):
        x = frames[:, s:s + chunk].permute(1, 0, 2, 3).float()
        r = ramp[s:s + chunk, None, None, None]
        m = mask[s:s + chunk]
        low = kornia.filters.gaussian_blur2d(x, (k, k), (sigma, sigma))
        ys = []
        for shift, alpha in corr:
            y = low + (1 - r * (1 - alpha)) * (x - low)
            lab = kornia.color.rgb_to_lab(((y + 1) / 2).clamp(0, 1)) + r * shift[None, :, None, None]
            ys.append(kornia.color.lab_to_rgb(lab).clamp(0, 1) * 2 - 1)
        out[:, s:s + chunk] = (m * ys[0] + (1 - m) * ys[1]).permute(1, 0, 2, 3).to(frames.dtype)
    info = {"person": {"shift": corr[0][0].tolist(), "detail_gain": corr[0][1].item()},
            "background": {"shift": corr[1][0].tolist(), "detail_gain": corr[1][1].item()}}
    return out, info
