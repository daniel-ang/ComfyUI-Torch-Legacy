import torch
from comfy_kitchen.backends.eager import quantization as q
from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation

_orig_int8_linear = q.int8_linear


def int8_linear(x, weight, weight_scale, bias=None, out_dtype=torch.bfloat16, convrot=False, convrot_groupsize=256,
                input_act=None, input_act_weight=None, input_act_eps=0.0, residual=None, residual_scale=None):
    # Same math as comfy_kitchen's eager int8_linear, but gives the int8 GEMM a column-major view of the weight
    # instead of a per-call weight.T.contiguous() copy. int32 accumulation is exact, so the output bits are unchanged.
    if input_act is not None or residual is not None or not x.is_cuda or weight.shape[0] % q._int8_mm_n_alignment(x):
        return _orig_int8_linear(x, weight, weight_scale, bias, out_dtype, convrot, convrot_groupsize,
                                 input_act, input_act_weight, input_act_eps, residual, residual_scale)
    weight = weight.to(device=x.device)
    weight_scale = weight_scale.to(device=x.device, dtype=torch.float32).reshape(1, -1)
    if convrot:
        x = _rotate_activation(x, _build_hadamard(convrot_groupsize, device=x.device, dtype=x.dtype), convrot_groupsize)
    orig_shape = x.shape
    x_8, x_scale = q.quantize_int8_rowwise(x.reshape(-1, x.shape[-1]))
    m, k = x_8.shape
    pad_m = max(32, (m + 31) // 32 * 32) - m
    if pad_m or k % 8:
        x_8 = torch.nn.functional.pad(x_8, (0, (-k) % 8, 0, pad_m))
        weight = torch.nn.functional.pad(weight, (0, (-k) % 8))
    result = torch._int_mm(x_8, weight.t())[:m]
    result = (result.float() * (x_scale.float() * weight_scale)).to(out_dtype)
    if bias is not None:
        result = result + bias.to(device=result.device, dtype=result.dtype).reshape(1, -1)
    return result.reshape(*orig_shape[:-1], result.shape[-1])
