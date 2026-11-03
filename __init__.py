"""Reproduce torch 2.8.0+cu129 numerics on newer torch, so MiniMax Music 3 / YuE2 seeds give the same songs.

Importing this changes nothing. The "Torch Legacy Numerics" node returns a MODEL and CLIP whose sampling and
encode/generate run with these process-wide patches, which are removed again afterwards:
- comfy-kitchen CUDA backend disabled (its fused kernels round differently from the eager path torch 2.8 used)
- F.rms_norm on CUDA summed in torch 2.8's reduction order
- Tensor.addcmul_ as multiply-then-add (torch 2.8 rounding, no FMA)
- eager int8_linear without the per-call weight transpose copy (bit-identical, faster)
"""
import contextlib
import logging

import torch
import comfy_kitchen
import comfy_kitchen.backends.eager as eager_backend
from comfy_kitchen.backends.eager import quantization
from typing_extensions import override

import comfy.model_prefetch
import comfy.patcher_extension
from comfy_api.latest import ComfyExtension, io

from . import fastint8
from .emu import rms_norm_28

_orig_rms_norm = torch.nn.functional.rms_norm
_orig_addcmul_ = torch.Tensor.addcmul_
_orig_int8_linear = quantization.int8_linear
_warned = set()
_depth = 0
_restore_kitchen_cuda = False


def rms_norm(input, normalized_shape, weight=None, eps=None):
    if input.is_cuda and eps is not None and len(normalized_shape) == 1:
        out = rms_norm_28(input, weight, eps)
        if out is not None:
            return out
    key = (input.shape[-1], input.device.type)
    if key not in _warned:
        _warned.add(key)
        logging.warning(f"Torch Legacy: rms_norm shape {tuple(input.shape)} on {input.device.type} not emulated, using native torch")
    return _orig_rms_norm(input, normalized_shape, weight, eps)


def addcmul_(self, tensor1, tensor2, *, value=1):
    if value != 1:
        return _orig_addcmul_(self, tensor1, tensor2, value=value)
    return self.add_(tensor1 * tensor2)


def _set_active(active):
    global _restore_kitchen_cuda
    # Captured CUDA graphs replay whatever numerics they were recorded with, so none may cross the switch.
    comfy.model_prefetch.cleanup_malloc_graph()
    if active:
        _restore_kitchen_cuda = comfy_kitchen.registry.is_available("cuda")
        comfy_kitchen.registry.disable("cuda")
    elif _restore_kitchen_cuda:
        _restore_kitchen_cuda = False
        comfy_kitchen.registry.enable("cuda")
    torch.nn.functional.rms_norm = rms_norm if active else _orig_rms_norm
    torch.Tensor.addcmul_ = addcmul_ if active else _orig_addcmul_
    quantization.int8_linear = eager_backend.int8_linear = fastint8.int8_linear if active else _orig_int8_linear


@contextlib.contextmanager
def legacy_numerics():
    global _depth
    _depth += 1
    try:
        if _depth == 1:
            _set_active(True)
        yield
    finally:
        _depth -= 1
        if _depth == 0:
            _set_active(False)


def _wrap(f):
    def wrapped(*args, **kwargs):
        with legacy_numerics():
            return f(*args, **kwargs)
    return wrapped


def _outer_sample(executor, *args, **kwargs):
    with legacy_numerics():
        return executor(*args, **kwargs)


class TorchLegacyNumerics(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TorchLegacyNumerics",
            display_name="Torch Legacy Numerics",
            category="advanced/debug/model",
            description="Runs the model's sampling and the CLIP's text encode/generate with torch 2.8.0 CUDA numerics, so MiniMax Music3 / YuE2 seeds reproduce songs made on torch 2.8.",
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
            ],
            outputs=[io.Model.Output(), io.Clip.Output()],
        )

    @classmethod
    def execute(cls, model, clip):
        model = model.clone()
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, "torch_legacy_numerics", _outer_sample)
        clip = clip.clone()
        for name in ("encode_token_weights", "generate"):
            clip.patcher.add_object_patch(name, _wrap(clip.patcher.get_model_object(name)))
        return io.NodeOutput(model, clip)


class TorchLegacyExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [TorchLegacyNumerics]


async def comfy_entrypoint():
    return TorchLegacyExtension()
