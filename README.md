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
g++ -std=c++20 -shared -fPIC -O1 -Iinclude examples/hello/hello_tog.cpp -o examples/hello/tog.so
build/bin/Simulator --config configs/tpuv6e.yml \
  --tog_so examples/hello/tog.so --cycle_table examples/hello/tog_cycles.tsv \
  --unit_table examples/hello/tog_units.tsv | tail -n 1
```

```text
[2026-10-12 14:29:12.768] [info] Total execution cycles: 2437
```

The kernel took 2437 cycles of the core clock on that machine.

## A whole TOG

`examples/hello/hello_tog.cpp`, the TOG of the Quick start:

```cpp
// The smallest TOG: one work-item loads a 64x64 f32 tile of tensor 0 into IN, computes OUT from it (tile 0 of the
// cycle table), and stores OUT to tensor 1. Each call is one modeled instruction; the buffers order them.
#include "togsim.hpp"
using namespace togsim;

struct Hello : Tog<Hello> {
  static constexpr Buf IN{.id = 1, .bytes = f32(64, 64)}, OUT{.id = 2, .bytes = f32(64, 64)};
  static constexpr std::array buffers{IN, OUT};
  static constexpr Node STEP{.id = 0, .count = 1};
  static constexpr Desc SRC{.ptr = 0, .type = f32, .offset = 0, .dims = {64, 64}, .strides = {64, 1}};
  static constexpr Desc DST{.ptr = 1, .type = f32, .offset = 0, .dims = {64, 64}, .strides = {64, 1}};

  void entry(Kernel& k) const { dispatch<&Hello::body>(k); }

  void body(Item& w) const {
    load(w, {.from = SRC, .to = IN});
    compute(w, {.node = STEP, .reads = {IN}, .writes = {OUT}});
    store(w, {.from = OUT, .to = DST});
  }
};
```

- **`buffers`** lists the spad buffers, each with its id and byte size.
- **`entry`** runs once and dispatches each work-item, the unit of parallelism: a body function
  bound to one core.
- **`body`** holds the instructions of the work-item: a load of tensor argument 0 into buffer 1, a
  compute that reads buffer 1 and writes buffer 2, and a store of buffer 2 to tensor argument 1.
  The compute takes row 0 of the cycle table (`STEP`, node 0 of 1).
- **The buffers order the instructions.** An instruction waits for the earlier instruction that
  wrote a buffer it reads, and for nothing else.

Three files go with it:

| File | Contents | Meaning |
|---|---|---|
| `tog_tensors.txt`, beside `tog.so` | `16384`<br>`16384` | The byte size of each tensor argument. |
| `tog_cycles.tsv`, `--cycle_table` | `600<TAB>0` | Compute tile 0 (`tile_id` 0) takes 600 cycles. |
| `tog_units.tsv`, `--unit_table` | `0<TAB>TPUV6E_MXU<TAB>rows<TAB>1<TAB>128`<br>`0<TAB>Vpu<TAB>cycles<TAB>1<TAB>100` | What each hardware unit did during one run of compute tile 0. |

[docs/tutorial.md](docs/tutorial.md) reads this TOG and its trace line by line, then builds up to
asynchronous loads, many work-items, the array split and sessions.

## Writing a TOG

A TOG is written against **TOGSim API v1.0.0**, the one header `include/togsim.hpp` (C++20), by
hand or by a compiler. Every call takes one struct of named arguments:

| Call | What it models |
|---|---|
| `dispatch<&X::f>(k, ...)` | One work-item, from the kernel's `entry`. |
| `load(w, {.from, .to, ...})`, `store(w, {.from, .to, ...})` | One DMA between DRAM and the scratchpad. |
| `load_async`, `semaphore`, `wait(w, {.sem})` | Asynchronous loads and the wait for them. |
| `compute(w, {.node, .variant, .reads, .writes})` | One compute. Its cycles are a row of the cycle table. |

The header lowers each call to the C ABI 16 of `include/togsim_runtime.h`, which is what TOGSim
loads. A body is called three times, so it must make the same calls every time, and it must call
the producer of a buffer before its consumer.

[docs/tog-api.md](docs/tog-api.md) is the API reference. [docs/tog-abi.md](docs/tog-abi.md) is
the C ABI under it: what each call models, the buffer and scratchpad rules, the files and the two
tables, and every refusal with its cause and fix.

## Examples

`examples/run.sh [name]...` builds and replays the examples and prints the lines of each log that
matter; `examples/<name>/expected.txt` is that output.

| Example | What it shows | Total cycles |
|---|---|---|
| `hello` | One load, one compute, one store. | 2437 |
| `async_barrier` | Two asynchronous loads and the barrier that waits for them. | 2479 |
| `tiles` | Eight work-items on one, two and four cores. | 8301, 4824, 3086 |
| `gemm` | Two kinds of compute, the unit table and the array split. | 2719 |
| `session` | Two kernels from a command stream, on one partition and on two. | 4916, 2483 |
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

The machine is the YAML file given to `--config`. `configs/` ships 7 of them, and
`examples/minimal.yml` is one that depends on no other file.

## Documentation

| Page | What it holds |
|---|---|
| [docs/tutorial.md](docs/tutorial.md) | The model, and the six examples step by step. |
| [docs/tog-api.md](docs/tog-api.md) | TOGSim API v1.0.0: the header every TOG is written against. |
| [docs/tog-abi.md](docs/tog-abi.md) | The C ABI under it: what each call models, the files, the tables and the refusals. |
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
