"""Compatibility shim for running on GPUs newer than the installed PyTorch build.

`torch.prod` is one of the few reductions that PyTorch compiles at run time through
nvrtc.  When the GPU's compute capability is newer than anything the PyTorch build
knows about, that compilation fails with

    nvrtc: error: invalid value for --gpu-architecture (-arch)

Everything else keeps working, because precompiled kernels are forward-compatible
through the driver's own PTX JIT -- only the run-time nvrtc path is affected.  With
torch 1.12.1+cu113 (arch list up to sm_86, bundled nvrtc 11.2) this shows up on Ada
cards such as the L40S (sm_89).

`cumprod` has a native kernel, and the last element of a cumulative product is the
full product, so `x.cumprod(dim).select(dim, -1)` is a drop-in replacement.  It is
bit-identical for integers and for reductions of length <= 2; for longer float
reductions it differs by the float32 epsilon (measured max relative error 5e-7 at
length 100), because `prod` reduces as a tree while `cumprod` is sequential.

The patch is installed only when the device is actually newer than the build, so on
supported hardware the original implementation is used and behaviour is unchanged.
"""

import torch

_orig_prod_function = torch.prod
_orig_prod_method = torch.Tensor.prod

_INT_PROMOTE = (torch.bool, torch.int8, torch.uint8, torch.int16, torch.int32)


def device_is_newer_than_build():
    """True when the current GPU's arch is newer than every arch PyTorch was built for."""
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    device_arch = major * 10 + minor
    build_archs = [int(a.split("_")[1]) for a in torch.cuda.get_arch_list() if a.startswith("sm_")]
    return bool(build_archs) and device_arch > max(build_archs)


def _prod_via_cumprod(input, dim=None, keepdim=False, dtype=None):
    x = input
    if dtype is not None:
        x = x.to(dtype)
    elif x.dtype in _INT_PROMOTE:
        x = x.to(torch.int64)  # matches torch.prod's own integer promotion

    if dim is None:
        x = x.flatten()
        if x.numel() == 0:
            return torch.ones((), dtype=x.dtype, device=x.device)
        return x.cumprod(0)[-1]

    if x.shape[dim] == 0:
        out = torch.ones_like(x.select(dim, 0)) if x.numel() else x.sum(dim)
    else:
        out = x.cumprod(dim).select(dim, -1)
    return out.unsqueeze(dim) if keepdim else out


def _prod_dispatch(input, *args, **kwargs):
    if torch.is_tensor(input) and input.is_cuda:
        return _prod_via_cumprod(input, *args, **kwargs)
    return _orig_prod_function(input, *args, **kwargs)


def _prod_method_dispatch(self, *args, **kwargs):
    if self.is_cuda:
        return _prod_via_cumprod(self, *args, **kwargs)
    return _orig_prod_method(self, *args, **kwargs)


def install(force=False, verbose=True):
    """Patch torch.prod when the GPU is newer than the PyTorch build. Returns True if patched."""
    if not (force or device_is_newer_than_build()):
        return False
    if getattr(torch.prod, "_macarons_cumprod_shim", False):
        return True

    _prod_dispatch._macarons_cumprod_shim = True
    torch.prod = _prod_dispatch
    torch.Tensor.prod = _prod_method_dispatch

    if verbose:
        major, minor = torch.cuda.get_device_capability()
        print(f"[macarons] torch.prod routed through cumprod: sm_{major}{minor} is newer than "
              f"this torch build (max {torch.cuda.get_arch_list()[-1]}).")
    return True
