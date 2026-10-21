import os
import torch
import math
from torch.nn.attention.flex_attention import flex_attention as flex_attention_func


# The generic varlen path (text/image cross-attention, reference pass) uses FlashAttention-2.
# FlashAttention-3 is built without varlen support and is only used for the main
# in-context self-attention (incontext_attention with WAN_ATTN=fa3).
try:
    from flash_attn import flash_attn_varlen_func

    FLASH_VER = 2
except ModuleNotFoundError:
    try:
        from flash_attn_interface import flash_attn_varlen_func

        FLASH_VER = 3
    except ModuleNotFoundError:
        flash_attn_varlen_func = None  # in compatible with CPU machines
        FLASH_VER = None

print(f"[PreInfo] Use flash attention={FLASH_VER}")

flex_attention_func = torch.compile(
    flex_attention_func,
    dynamic=False,             
    mode="max-autotune",
    fullgraph=True,     
    backend="inductor"
)

__all__ = [
    'flash_attention',
    'flex_attention'
]

def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == "cuda" and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor([lq] * b, dtype=torch.int32).to(
            device=q.device, non_blocking=True
        )
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor([lk] * b, dtype=torch.int32).to(
            device=k.device, non_blocking=True
        )
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale
    # apply attention
    if FLASH_VER == 3:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        )[0].unflatten(0, (b, lq))
    else:
        assert FLASH_VER == 2
        x = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
        ).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


# Author: Guangyuan Wang
def flex_attention(
    q, 
    k, 
    v,
    q_lens=None,
    k_lens=None, 
    block_mask=None, 
    kernel_options=None,
    dtype=torch.bfloat16,
    score_mod=None
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda'
    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype
    def half(x): return x if x.dtype in half_dtypes else x.to(dtype)

    assert lq % 128 == 0, "q_len must be divisible by 128."
    assert lk % 128 == 0, "k_len must be divisible by 128."

    # preprocess query
    if q_lens is None:
        q = half(q)
    else:
        q = half(q)
        assert q_lens.max() == q_lens.min(), 'varlen of query is not supported'

    # preprocess key, value
    if k_lens is None:
        k, v = half(k), half(v)
    else:
        k, v = half(k), half(v)
        assert k_lens.max() == k_lens.min(), 'varlen of key is not supported'

    q = q.to(v.dtype)
    k = k.to(v.dtype)


    x = flex_attention_func(
        query=q.transpose(2,1),
        key=k.transpose(2,1),
        value=v.transpose(2,1),
        block_mask=block_mask,
        kernel_options=kernel_options,
        score_mod=score_mod
    ).transpose(2, 1)
    
    return x.type(out_dtype)

def _fp8(x):
    """Per-(batch, head) scaled float8_e4m3 copy of a [B, L, N, C] tensor and its descale [B, N]."""
    descale = x.abs().amax(dim=(1, 3)).float().clamp_min(1e-6) / 448.0
    x8 = (x.float() / descale[:, None, :, None]).to(torch.float8_e4m3fn)
    return x8, descale


def _attn_with_lse(q, k, v, descale=None):
    """Dense attention on [B, L, N, C] tensors -> (out [B, Lq, N, C], lse [B, Lq, N], natural log).

    WAN_ATTN=fa3 uses FlashAttention-3 (Hopper); otherwise FlashAttention-2. With
    `descale` = (q_descale, k_descale, v_descale), q/k/v are float8 (FA3 FP8 kernel).
    """
    if descale is not None:
        from flash_attn_interface import _flash_attn_forward
        out, lse, *_ = _flash_attn_forward(
            q, k, v, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, *descale,
            q.shape[-1] ** -0.5, False,
        )
        return out, lse.transpose(1, 2)
    if os.environ.get("WAN_ATTN") == "fa3":
        # the public flash_attn_func of FA3 does not return the LSE; call the forward op directly
        from flash_attn_interface import _flash_attn_forward
        out, lse, *_ = _flash_attn_forward(
            q, k, v, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None,
            q.shape[-1] ** -0.5, False,
        )
        return out, lse.transpose(1, 2)
    from flash_attn import flash_attn_func
    out, lse, _ = flash_attn_func(q, k, v, return_attn_probs=True)
    return out, lse.transpose(1, 2)


def incontext_attention(q, k, v, hw, q_limit, q_total, k_limit, log_scale, out_dtype=None):
    """Exact equivalent of the flex_attention call in Block.forward_gen.

    The flex block mask lets every valid query attend to (a) all generated tokens
    [0, q_limit) and (b) the driving-video tokens of its own frame (key frame j of
    the reference part belongs to query frame j + 1); score_mod adds `log_scale`
    to the logits of generated frame 1 (keys [hw, 2hw)). Instead of a masked
    kernel over the padded layout, attend to each key group with dense flash
    attention and merge the partial results through their log-sum-exp.

    q: [B, Lq_pad, N, C]; k, v: [B, q_total + Lref_pad, N, C].
    Returns [B, Lq_pad, N, C] (rows >= q_limit are zero, as with the masked kernel).
    """
    out_dtype = out_dtype or q.dtype
    # rope_apply returns float32; like flex_attention's wrapper, run the kernel in bf16
    q, k, v = (x if x.dtype in (torch.float16, torch.bfloat16) else x.to(torch.bfloat16) for x in (q, k, v))
    B, Lq_pad, N, C = q.shape
    out_full = torch.zeros(q.shape, device=q.device, dtype=out_dtype)
    n_frames = q_limit // hw
    n_ref_frames = k_limit // hw
    dev = q.device

    fp8 = os.environ.get("WAN_ATTN") == "fa3fp8"
    for b in range(B):
        qb = q[b, :q_limit]
        kb, vb = k[b], v[b]
        if fp8:
            # FP8 (e4m3) attention, per-head scales. Keys are first centred by their per-head
            # mean over the generated tokens: subtracting the same vector from every key shifts
            # all logits of a query by the same constant (and every group's LSE equally), so the
            # merged softmax is unchanged, but the quantization range shrinks.
            kb = kb - kb[:q_limit].float().mean(0, keepdim=True).to(kb.dtype)
            qb, dq = _fp8(qb[None])
            kb, dk = _fp8(kb[None])
            vb, dv = _fp8(vb[None])
            qb, kb, vb = qb[0], kb[0], vb[0]

        def attn(qx, kx, vx):
            if not fp8:
                return _attn_with_lse(qx, kx, vx)
            nb = qx.shape[0]
            return _attn_with_lse(qx, kx, vx, (dq.expand(nb, -1).contiguous(), dk.expand(nb, -1).contiguous(),
                                               dv.expand(nb, -1).contiguous()))
        # Streaming log-sum-exp merge (online softmax): one fp32 accumulator, each partial
        # attention folded in as soon as it is computed, to keep peak memory low at 1080p.
        acc = torch.zeros(q_limit, N, C, device=dev, dtype=torch.float32)
        run_m = torch.full((q_limit, N), float("-inf"), device=dev)
        run_s = torch.zeros(q_limit, N, device=dev)

        def fold(rows, o, l):
            m_old = run_m[rows]
            m_new = torch.maximum(m_old, l)
            scale_old = torch.exp(m_old - m_new)
            scale_new = torch.exp(l - m_new)
            acc[rows] = acc[rows] * scale_old[..., None] + o.float() * scale_new[..., None]
            run_s[rows] = run_s[rows] * scale_old + scale_new
            run_m[rows] = m_new

        all_rows = slice(0, q_limit)
        # (a) generated tokens: frame 0 keys, frame 1 keys (+ logit bias), frames 2.. keys
        for k0, k1, bias in ((0, hw, 0.0), (hw, 2 * hw, log_scale), (2 * hw, q_limit, 0.0)):
            if k1 <= k0:
                continue
            o, l = attn(qb[None], kb[k0:k1][None], vb[k0:k1][None])
            fold(all_rows, o[0], l[0].float() + bias)
            del o, l
        # (b) query frame j+1 attends driving frame j only: a batch of per-frame attentions
        n_cond = min(n_frames - 1, n_ref_frames)
        if n_cond > 0:
            rows = slice(hw, (n_cond + 1) * hw)
            qc = qb[rows]
            kc = kb[q_total:q_total + n_cond * hw]
            vc = vb[q_total:q_total + n_cond * hw]
            oc, lc = attn(*(t.reshape(n_cond, hw, N, C) for t in (qc, kc, vc)))
            fold(rows, oc.reshape(n_cond * hw, N, C), lc.reshape(n_cond * hw, N).float())
            del oc, lc
        out_full[b, :q_limit] = (acc / run_s[..., None]).to(out_dtype)
        del acc, run_m, run_s
    return out_full
