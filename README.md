# ComfyUI Torch Legacy

Newer PyTorch releases (2.9+) changed the summation order of some CUDA reductions, and ComfyUI's comfy-kitchen fused kernels round differently from the eager path torch 2.8 used. The result: the same MiniMax Music 3 / YuE2 workflow and seed produces a different song after upgrading torch.

This node restores torch 2.8.0+cu129 numerics on newer torch so old seeds reproduce.

## Node

**Torch Legacy Numerics** (`advanced/debug/model`)

Put it between your loader and the sampler / text encode:

- `model` -> sampling runs with torch 2.8 numerics
- `clip` -> text encode / generate runs with torch 2.8 numerics

Installing the node changes nothing on its own. The patches below are applied only while the patched model samples or the patched CLIP encodes/generates, and removed afterwards:

- comfy-kitchen CUDA backend disabled
- `F.rms_norm` on CUDA summed in torch 2.8's reduction order
- `Tensor.addcmul_` as multiply-then-add (no FMA)
- eager `int8_linear` without the per-call weight transpose copy (bit-identical, faster)

## Install

ComfyUI-Manager: **Install via Git URL** with this repo's URL, or:

```
cd ComfyUI/custom_nodes
git clone https://github.com/daniel-ang/ComfyUI-Torch-Legacy
```

No extra pip packages are needed.

## Compatibility

- Tested with ComfyUI 0.43.0, comfy-kitchen 0.2.40, torch 2.13.0+cu130.
- Uses private comfy-kitchen internals for the int8 path, so a comfy-kitchen update can break it.
- CUDA only. `rms_norm` shapes that aren't emulated fall back to native torch with a one-time warning, and those results won't match torch 2.8 exactly.

## License

GPL-3.0
