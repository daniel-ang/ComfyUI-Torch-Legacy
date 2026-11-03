import torch


def _last_pow2(n):
    return 1 << (n.bit_length() - 1)


def mean_sq_28(x2):
    """Row mean of a contiguous fp32 [R, D] CUDA tensor, summed in the exact order of torch 2.8's
    CUDA reduce kernel (Reduce.cuh, vt0=4, input_vec_size=4, warp shuffle offsets 1->16).
    torch 2.9+ reversed the warp shuffle order. Returns None for configs that aren't emulated."""
    R, D = x2.shape
    vec = D > 128
    dim0 = D // 4 if vec else D
    p0 = 512 if dim0 >= 512 else _last_pow2(dim0)
    p1 = 512 if R >= 512 else _last_pow2(R)
    bw = min(p0, 32)
    bh = min(p1, 512 // bw)
    bw = min(p0, 512 // bh)
    if -(-D // bw) >= min(bh * 16, 256) or (vec and D % 4):
        return None
    # The kernel's partial last chunk is equivalent to reducing zeros (squares are >= 0, 0 + v is exact).
    pad = (-D) % (4 * bw)
    if pad:
        x2 = torch.nn.functional.pad(x2, (0, pad))
    chunks = (D + pad) // (4 * bw)
    if vec:
        v = x2.view(R, chunks, bw, 4)
    else:
        v = x2.view(R, chunks, 4, bw).transpose(-1, -2)
    # per-thread accumulators, one fp32 add per loop iteration
    acc = v[:, 0]
    for k in range(1, v.shape[1]):
        acc = acc + v[:, k]
    a = ((acc[..., 0] + acc[..., 1]) + acc[..., 2]) + acc[..., 3]
    # shared memory stage (block_x_reduce, offsets bw/2 down to 32)
    off = bw // 2
    while off >= 32:
        a = a[:, :off] + a[:, off:2 * off]
        off //= 2
    # warp shuffle with increasing offsets: lane 0 ends up with an adjacent-pair tree
    while a.shape[1] > 1:
        a = a[:, 0::2] + a[:, 1::2]
    return a * (1.0 / D)


def rms_norm_28(x, weight, eps):
    """torch 2.8 CUDA F.rms_norm: fp32 mean of squares, rsqrt, multiply (weight in fp32). None if not emulated."""
    D = x.shape[-1]
    xf = x.float()
    ms = mean_sq_28((xf * xf).reshape(-1, D))
    if ms is None:
        return None
    out = xf * torch.rsqrt(ms.view(*x.shape[:-1], 1) + eps)
    if weight is not None:
        out = out * weight.float()
    return out.to(x.dtype)
