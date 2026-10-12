"""Specialise a kernel of kernels/ without touching its source: its tile and hardware constants are -D macros
at compile time, its dimensions the shape arguments TOGSim hands togsim_kernel from tog_shape.txt."""
from __future__ import annotations

import os
import shutil
import subprocess

from .onnx_ops import KernelRef


def defines(kernel: KernelRef, values: dict[str, int]) -> list[str]:
    """`NAME=value` for each tile and hardware macro of `kernel`; a name `values` lacks is refused, as the kernel
    would silently keep its default."""
    return [f"{name}={int(_value(kernel, values, name))}" for name in kernel.tile_names + kernel.hw_names]


def shape_args(kernel: KernelRef, values: dict[str, int]) -> list[int]:
    """The kernel's shape arguments, in the order it reads them."""
    return [int(_value(kernel, values, name)) for name in kernel.size_names]


def _value(kernel: KernelRef, values: dict[str, int], name: str) -> int:
    if name not in values:
        raise KeyError(f"{kernel.stem}: no value for {name!r}; the names in onnx_ops.py no longer match shapes.py")
    return values[name]


def compile_so(cpp_path: str, include_dir: str, out_so: str, macros: list[str] = ()) -> str:
    """Compile a kernel to `out_so` with -D<macro> each; an up-to-date `out_so` is kept."""
    if os.path.exists(out_so) and \
            os.path.getmtime(out_so) >= os.path.getmtime(cpp_path):
        return out_so
    os.makedirs(os.path.dirname(out_so) or ".", exist_ok=True)
    subprocess.run(
        [os.environ.get("CXX", "g++"), "-shared", "-fPIC", "-std=gnu++17", "-O1",
         "-I", include_dir, *[f"-D{m}" for m in macros], cpp_path, "-o", out_so],
        check=True, capture_output=True, text=True, errors="replace")
    return out_so


def write_tog_dir(tog_dir: str, so: str, shape: list[int], tensors: list[int]) -> str:
    """Lay out one kernel run as TOGSim reads it: tog.so with tog_shape.txt and tog_tensors.txt beside it."""
    os.makedirs(tog_dir, exist_ok=True)
    tog = os.path.join(tog_dir, "tog.so")
    shutil.copyfile(so, tog)
    with open(os.path.join(tog_dir, "tog_shape.txt"), "w") as fh:
        fh.write("".join(f"{v}\n" for v in shape))
    with open(os.path.join(tog_dir, "tog_tensors.txt"), "w") as fh:
        fh.write("".join(f"{b}\n" for b in tensors))
    return tog
