# The kernel library

`kernels/` holds fourteen kernels written by hand against the [TOG ABI](tog-abi.md), one
`<name>_tog.cpp` each: `gemm`, `conv`, `attention`, `softmax`, `layernorm`, `embed_layernorm`,
`global_avgpool`, `bias_act`, `bias_gelu`, `maxpool`, `adaptive_avgpool`, `concat`,
`kvcache_concat` and `flatten`. They come from
[PyTorchSim PR #315](https://github.com/PSAL-POSTECH/PyTorchSim/pull/315) by Lee JaeGyun, ported
from TOG ABI 12 to 16. `tests/kernels/` replays them, and the [frontend](frontend.md) builds on
them; neither the simulator nor its build needs them.

- **Shape.** A kernel's dimensions are the shape arguments of `togsim_kernel`, which TOGSim reads
  from `tog_shape.txt` beside the TOG. The comment at the head of each file names them in order
  (`gemm`: `M K N`). Without a `tog_shape.txt` a kernel runs its example shape; another count
  than its own, or a dimension below 1, ends the run with exit `1` and a `kernel <name>: ...` line
  on stderr.
- **Convolution.** `conv` takes sixteen: `I_C I_H I_W O_C O_H O_W K_H K_W`, the strides `S_H S_W`,
  the dilations `D_H D_W` and the four pads `P_T P_L P_B P_R` (top, left, bottom, right, ONNX's
  order; these may be 0). Its input tensor is the padded input, `(I_H + P_T + P_B) x
  (I_W + P_L + P_R)` pixels of `I_C` channels, and output `(oh, ow)` reads row `oh*S_H + kh*D_H`
  and column `ow*S_W + kw*D_W` of it at kernel position `(kh, kw)`. For a kernel position a tile
  loads the one input pixel under each of its outputs, so a stride changes the addresses of the
  loads and not their number or size. The addresses need the pads only as the padded extent, so
  an asymmetric padding costs nothing extra. `O_H` and `O_W` are arguments, not derived: a shape
  whose last output would read past the padded input is refused (exit `1`). It has no group count.
- **Tile.** The tile a work-item moves and computes is compile-time, a `-D` macro per extent with
  a default (`gemm`: `-DTM=32 -DTN=256 -DTK=64`). It cannot be a shape argument: TOGSim reads
  `togsim_spad_buffer_bytes` before it calls `togsim_kernel`, and the buffer sizes follow from the
  tile. A dimension the tile does not divide still gets a whole last tile: the kernel moves and
  computes past the end of the tensor there, so the shapes to trust are multiples of the tile.
- **Compute.** Every `togsim_compute` is `compute_type` `0`. `gemm`, `conv` and `attention` issue
  one compute per block of systolic work (a `TM x TK x TN` block of gemm's K loop) instead of one
  per preload and matmul; what a block costs and how many Systolic rows it holds is the cycle
  table's and the unit table's to say.

## Replaying a kernel by hand

Compile it, put `tog_tensors.txt` beside it (one byte size per tensor
argument) and give the simulator a cycle table and a unit table with a row per tile it computes:

```bash
mkdir -p /tmp/gemm && g++ -std=c++17 -shared -fPIC -Iinclude kernels/gemm_tog.cpp -o /tmp/gemm/tog.so
printf '131072\n262144\n131072\n' > /tmp/gemm/tog_tensors.txt        # A 128x256, B 256x256, C 128x256
printf '128\t0\n8206\t0\n' > /tmp/gemm/tog_cycles.tsv                # tile 0 accumulator init, tile 1 a block
printf '0\tSystolic\trows\t1\t0\n1\tSystolic\trows\t1\t8192\n' > /tmp/gemm/tog_units.tsv
build/bin/Simulator --config configs/systolic_ws_256x256_c1_simple_noc_tpuv6e.yml \
  --tog_so /tmp/gemm/tog.so --cycle_table /tmp/gemm/tog_cycles.tsv --unit_table /tmp/gemm/tog_units.tsv
```
