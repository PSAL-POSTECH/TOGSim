# The kernel library

`kernels/` holds fourteen kernels written by hand against [TOGSim API 1.0.0](tog-api.md), one
`<name>_tog.cpp` each: `gemm`, `conv`, `attention`, `softmax`, `layernorm`, `embed_layernorm`,
`global_avgpool`, `bias_act`, `bias_gelu`, `maxpool`, `adaptive_avgpool`, `concat`,
`kvcache_concat` and `flatten`. They come from
[PyTorchSim PR #315](https://github.com/PSAL-POSTECH/PyTorchSim/pull/315) by Lee JaeGyun.
`tests/kernels/` runs them, and the [frontend](frontend.md) builds on
them; neither the simulator nor its build needs them.

- **Shape.** A kernel's dimensions are its shape arguments (`shape<N>`), which TOGSim reads
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
- **Tile.** The tile a work-item moves and computes is compile-time, a `-D` build option per
  extent with a default (`gemm`: `-DTM=32 -DTN=256 -DTK=64`). It cannot be a shape argument: TOGSim
  reads the buffer sizes before it runs the kernel's `entry`, and they follow from the tile.
- **Short last tiles.** A dimension the tile does not divide ends in a short tile: its DMAs move
  what is left of the dimension and no DMA leaves its tensor (conv clips its `SA`-wide loads the
  same way, so 3 input channels are loaded as 3). A short tile costs something else, so it computes
  under its own [variant](tog-api.md#rows-node-and-variant): bit `i` is set when the tile is short
  on the kernel's axis `i` (`short_axis`); variant 0 is the whole tile. A compute outside an inner loop
  carries only the axes of the loops around it. The axes, in bit order:

  | Kernel | Nodes | Axes (bit 0 first) |
  |---|---|---|
  | `gemm` | 2 | `M`, `N`, `K` (acc_init: `M`, `N`) |
  | `conv` | 2 | `O_C`, `O_H`, `O_W`, `I_C` (acc_init: the first three) |
  | `attention` | 9 | `SEQ` by `Q_LEN`, `SEQ` by `KV_BLOCK` (normalize: the first) |
  | `softmax` | 5 | `M`, `N` (the two write nodes: `M`) |
  | `layernorm` | 4 | `M` by `TM`, `N`, `M` by `TM_STATS` (statistics: the last two, its writes the last; normalize: the first two) |
  | `global_avgpool` | 3 | `C` by `TC`, `HW`, `C` by `TC_SCALE` (sum: the first two, its write the first; scale: the last) |
  | `bias_act`, `bias_gelu` | 1 | `M`, `N` |
  | `embed_layernorm` | 1 | `SEQ`, `DIM` |
  | `maxpool`, `adaptive_avgpool` | 1 | `ROWS`, `COLS` |
  | `kvcache_concat` | 1 | `TOKENS` |
  | `concat`, `flatten` | 0 | no compute; `concat` clips its DMAs |

  A gemm of `M K N = 100 200 300` in the default tile computes acc_init as ids 0, 2, 4, 6 and a
  block as the odd ids 1 to 15; a cycle table for it has 16 lines.
- **Compute.** `gemm`, `conv` and `attention` issue
  one compute per block of systolic work (a `TM x TK x TN` block of gemm's K loop) instead of one
  per preload and matmul; what a block costs and how many array-unit rows it holds is the cycle
  table's and the unit table's to say.

## Running a kernel by hand

Compile it, put `tog_tensors.txt` beside it (one byte size per tensor
argument) and give the simulator a cycle table and a unit table with a row per tile id it computes
(two at a shape the tile divides, as here; the ids of the short variants as well at another):

```bash
mkdir -p /tmp/gemm && g++ -std=c++20 -shared -fPIC -Iinclude kernels/gemm_tog.cpp -o /tmp/gemm/tog.so
printf '131072\n262144\n131072\n' > /tmp/gemm/tog_tensors.txt        # A 128x256, B 256x256, C 128x256
printf '128\t0\n8206\t0\n' > /tmp/gemm/tog_cycles.tsv                # tile 0 accumulator init, tile 1 a block
printf '0\tTPUV6E_MXU\trows\t1\t0\n1\tTPUV6E_MXU\trows\t1\t8192\n' > /tmp/gemm/tog_units.tsv
build/bin/Simulator --config configs/tpuv6e.yml \
  --tog_so /tmp/gemm/tog.so --cycle_table /tmp/gemm/tog_cycles.tsv --unit_table /tmp/gemm/tog_units.tsv
```
