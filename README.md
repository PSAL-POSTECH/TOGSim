# TOGSim

TOGSim tells you how many cycles a kernel takes on a modeled accelerator. Its input is a
tile-operation-graph (TOG): a small shared object whose function calls are the kernel's memory
transfers, computes and waits. A compiler emits a TOG, or you write one by hand in C++. TOGSim
loads it, replays those calls on a machine you describe in a YAML file (cores with systolic arrays
and a vector unit, DRAM, an interconnect), and reports the total cycles, how busy each unit was and
how much DRAM bandwidth the kernel used.

A TOG moves no data and contains no timing. What one compute costs comes from a table you supply.
The calls are the nodes of the graph, and the scratchpad buffers they read and write are its edges.

```
  a compiler, or you by hand
            |
            v
  tog.so            the TOG: calls only
  tog_tensors.txt   bytes of each tensor argument  --+
  tog_cycles.tsv    cycles of each kind of compute   +-->  build/bin/Simulator  -->  Total execution cycles: N
  tog_units.tsv     what each unit did per compute --+            ^                  per-core unit utilization
                                                                  |                  DMA and DRAM bandwidth
                                                   machine config (configs/*.yml)
```

## Quick start

TOGSim builds on Ubuntu 22.04 with GCC 11, `conan` 1.x and `cmake` 3.16 or newer, and needs
network access while it builds. On another system, work in a container
(`docker run --rm -it -v "$PWD":"$PWD" -w "$PWD" ubuntu:22.04 bash`); the directory is mounted at
its own path because the build tree records absolute paths.

```bash
apt-get update
apt-get install -y --no-install-recommends g++ make git python3 python3-pip ca-certificates
python3 -m pip install conan==1.56.0 cmake==3.26.4
```

```bash
git clone --recursive https://github.com/PSAL-POSTECH/TOGSim.git
cd TOGSim
```

```bash
conan profile new default --detect --force
conan profile update settings.compiler.libcxx=libstdc++ default
mkdir -p build && cd build
conan install .. --build=missing
cmake ..
make -j"$(nproc)"
cd ..
```

The build has succeeded when `build/bin/Simulator` exists. Now compile a TOG with plain `g++` and
replay it:

```bash
g++ -std=c++17 -shared -fPIC -O1 -Iinclude examples/hello/hello_tog.cpp -o examples/hello/tog.so
build/bin/Simulator --config configs/systolic_ws_256x256_c1_simple_noc_tpuv6e.yml \
  --tog_so examples/hello/tog.so --cycle_table examples/hello/tog_cycles.tsv \
  --unit_table examples/hello/tog_units.tsv | tail -n 1
```

```text
[2026-10-12 14:29:12.768] [info] Total execution cycles: 964
```

The kernel took 964 cycles of the core clock on that machine.

## A whole TOG

`examples/hello/hello_tog.cpp`, the TOG of the Quick start:

```cpp
// The smallest TOG: one work-item loads a 64x64 tile of 32-bit elements from tensor 0 into spad buffer 1,
// computes buffer 2 from it (tile 0 of the cycle table), and stores buffer 2 to tensor 1.
#include "togsim_runtime.h"

#define ROWS 64
#define COLS 64

// The body of the work-item: each call is one modeled instruction, and the buffer ids order them.
static void body(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)iv; (void)n_iv;
  const int64_t dims[2] = {ROWS, COLS}, strides[2] = {COLS, 1};
  const int64_t in[1] = {1}, out[1] = {2};
  togsim_dma(ctx, TOGSIM_DMA_LOAD, 0, 0, 2, dims, strides, 32, 0, 0, nullptr, 0, in, 1, 0);
  togsim_compute(ctx, 0, 0, 0, nullptr, in, 1, out, 1);
  togsim_dma(ctx, TOGSIM_DMA_STORE, 1, 0, 2, dims, strides, 32, 0, 0, out, 1, nullptr, 0, 0);
}

// The entry point, called once when the TOG is loaded: it registers the one work-item.
extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n_shape_args) {
  (void)shape_args; (void)n_shape_args;
  togsim_dispatch(ctx, body, nullptr, 0, 0);
}

// The byte size of each spad buffer id named above; id 0 is not a buffer.
extern "C" const int32_t togsim_spad_buffer_count = 3;
extern "C" const int64_t togsim_spad_buffer_bytes[3] = {0, ROWS * COLS * 4, ROWS * COLS * 4};
```

- **`togsim_kernel`** is the entry point. It runs once and calls `togsim_dispatch` once per
  work-item, the unit of parallelism: a body function bound to one core.
- **`body`** holds the instructions of the work-item: a load of tensor argument 0 into buffer 1, a
  compute that reads buffer 1 and writes buffer 2, and a store of buffer 2 to tensor argument 1.
- **The buffer ids order the instructions.** An instruction waits for the earlier instruction that
  wrote a buffer it reads, and for nothing else.
- **The buffer table** at the bottom gives the byte size of each buffer id.

Three files go with it:

| File | Contents | Meaning |
|---|---|---|
| `tog_tensors.txt`, beside `tog.so` | `16384`<br>`16384` | The byte size of each tensor argument. |
| `tog_cycles.tsv`, `--cycle_table` | `600<TAB>0` | Compute tile 0 (`tile_id` 0) takes 600 cycles. |
| `tog_units.tsv`, `--unit_table` | `0<TAB>Systolic<TAB>rows<TAB>1<TAB>256`<br>`0<TAB>Vpu<TAB>cycles<TAB>1<TAB>100` | What each hardware unit did during one run of compute tile 0. |

[docs/tutorial.md](docs/tutorial.md) reads this TOG and its trace line by line, then builds up to
asynchronous loads, many work-items, the array split and sessions.

## The TOG ABI

The ABI is one header, `include/togsim_runtime.h`, at version 16. A TOG includes it, defines
`togsim_kernel` and the buffer table, and calls four functions that TOGSim provides when it loads
the TOG:

| Call | What it models |
|---|---|
| `togsim_dispatch(ctx, fn, iv, n_iv, key)` | Registers one work-item. Called from `togsim_kernel`. |
| `togsim_dma(ctx, ...)` | One load or store between DRAM and the scratchpad, synchronous or asynchronous. |
| `togsim_compute(ctx, tile_id, ...)` | One compute. Its cycles are the row `tile_id` of the cycle table. |
| `togsim_memory_barrier(ctx, sem, expected)` | A wait for the asynchronous loads signaled on `sem`. |

Three rules that nothing checks for you:

- A body is called three times, so it must make the same calls every time: no counters, no I/O.
- Call the producer of a buffer before its consumer. Instructions that share no buffer are not
  ordered.
- Make body functions `static`.

[docs/tog-abi.md](docs/tog-abi.md) is the reference: every argument of every call, the buffer and
scratchpad rules, the files and the two tables, every refusal with its cause and fix, and a
checklist for a TOG generator.

## Examples

`examples/run.sh [name]...` builds and replays the examples and prints the lines of each log that
matter; `examples/<name>/expected.txt` is that output.

| Example | What it shows | Total cycles |
|---|---|---|
| `hello` | One load, one compute, one store. | 964 |
| `async_barrier` | Two asynchronous loads and the barrier that waits for them. | 1001 |
| `tiles` | Eight work-items on one, two and four cores. | 5124, 2775, 1696 |
| `gemm` | Two kinds of compute, the unit table and the array split. | 1241 |
| `session` | Two kernels from a command stream, on one partition and on two. | 1857, 908 |
| `mistakes` | What TOGSim refuses, what it warns about, and what it does not notice. | |

## Running

```text
build/bin/Simulator --config <yml> --tog_so <so> --cycle_table <tsv> [--unit_table <tsv>] [--log_level <level>]
build/bin/Simulator --config <yml> --models_list <file> [--log_level <level>]
```

The first form replays one TOG; the second replays a command stream that launches several. The
result is the last line of the log, `Total execution cycles: N`; the lines above it give the
utilization of each unit per core and the DMA and DRAM bandwidth. `--log_level trace` prints when
each instruction was issued and finished. A refused run exits non-zero with a line that names the
cause.

The machine is the YAML file given to `--config`. `configs/` ships 19 of them, and
`examples/minimal.yml` is one that depends on no other file.

## Documentation

| Page | What it holds |
|---|---|
| [docs/tutorial.md](docs/tutorial.md) | The model, and the six examples step by step. |
| [docs/tog-abi.md](docs/tog-abi.md) | The TOG ABI reference. |
| [docs/running.md](docs/running.md) | The command line, the command stream, how to read the output, troubleshooting and known limitations. |
| [docs/machine.md](docs/machine.md), [docs/config-keys.md](docs/config-keys.md) | The knobs of a machine config, and how to change and sweep them. Every machine config key. |
| [docs/output-lines.md](docs/output-lines.md) | A whole log line by line, and the lines a script may rely on. |
| [docs/compatibility.md](docs/compatibility.md) | What a compiler or a script may depend on, and how another repository pins TOGSim. |
| [docs/kernels.md](docs/kernels.md), [docs/frontend.md](docs/frontend.md) | The kernel library: fourteen hand-written kernels, their shape arguments and their tiles. The optional frontend layer: ACS, an analytic cycle model, and the ONNX frontend. |
| [docs/glossary.md](docs/glossary.md) | The terms. |

## Testing

```bash
python3 tests/run.py
```

It replays hand-written TOGs against `build/` and ends with `ALL PASS (N cases in ...)`.
[tests/README.md](tests/README.md) says what each directory pins and how to add a case.

## License

MIT; see [LICENSE](LICENSE). The submodules carry their own terms: `extern/ramulator2` in its
`LICENSE` file, `extern/booksim` in the header of each source file.
