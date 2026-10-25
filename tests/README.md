# Tests

`tests/run.py` runs every case of every `tests/*/cases.py` against a built tree and prints one
`PASS` or `FAIL` line per case. It needs `g++` with C++20 (g++ 10 or newer), `python3` (standard library only) and the default
build target.

```bash
python3 tests/run.py                 # every case, against build/
python3 tests/run.py -k examples/    # the cases whose id contains a string
python3 tests/run.py --list          # the case ids, with one line per directory
```

It exits 0 when all cases pass, 1 if a case failed, and 2 if it could not run (no build, or a `-k`
that selects nothing). A failing case says what it expected, what it saw and where its log is.

`--workdir <dir>` keeps every case's files (the TOG, its tables, the machine config, `sim.log`) in
`<dir>/<case id>/`. That makes any case an example to rerun by hand: the first line of its
`sim.log` is its command line.

## What the cases pin

| Directory | What its cases pin |
|---|---|
| `tests/abi/` | A TOG built against another ABI version is refused; the same TOG at this version runs. On the raw C ABI, a TOG without `togsim_kernel` or without the runtime attribute table is refused and a DMA with null strides runs unstrided. |
| `tests/array_split/` | The array split of a segment TOG, and what it refuses. |
| `tests/configs/` | Every shipped config runs a small TOG, and is refused without a unit table. Every one follows tpuv6e's convention: an `isa` with exactly one `pytorchsim_*` entry, `vcix`, `address_map` and `systolic_array_unit`. |
| `tests/cycle_table/` | A run without a cycle table, or with a `tile_id` the table does not hold, is refused. |
| `tests/cycles/` | `Total execution cycles` of four hand-written TOGs on tpuv6e, as goldens; three of the TOGs are here. |
| `tests/examples/` | `examples/run.sh <name>` prints `examples/<name>/expected.txt`; the fenced blocks of the READMEs and of `docs/` quote only lines of those files and of the examples' sources; every row of the README's refusal table is a refusal the mistakes example prints. |
| `tests/kernel_order/` | Three kernels of one stream, each loading what the one before stored, on one core and on two: no load of an address enters the memory partition before the store to it, nor the DRAM model before the store's write is acked, and a kernel's completion line is not before its last store did. Two kernels that end with a compute each log their completion before the `DEVICE_SYNC` after them returns. |
| `tests/kernels/` | Each kernel of [`kernels/`](../docs/kernels.md) at its example shape: its `MOVIN`/`MOVOUT`/`COMP`/`MEMORY_BAR` counts and its `Total execution cycles`, as goldens; that gemm, conv and attention read their shape arguments; that conv, at a stride, a dilation and symmetric or asymmetric pads (a 1x1 at stride 2, a 7x7 at stride 2 with pad 3 over 3 channels among them), moves exactly the input, weight and output elements a convolution of that shape reads and writes, read from the address, size and stride of every DMA; each kernel at a shape none of its tiles divides (and conv short on every axis, and with 112 output columns in tiles of 32): no DMA ends past its tensor, every element is moved as often as the shape says, the computes name the tile ids of the short variants and no other, and the counts and the total are goldens; the shape arguments a kernel refuses (exit `1`). No run that checks instruction counts logs a warn-level line: the tables and `tog_tensors.txt` cover every tile id and `arg_id` the kernel names. |
| `tests/outputs/` | The shape of the lines a run logs, against golden files. |
| `tests/result/` | `--result <json>` holds each value its end-of-run line prints (units with and without a unit table, the DMA, the DRAM channels and their combination, the total), and in a session each kernel's completion and the wall clock; no `--result`, no file; a path that cannot be written is refused, exit 1. |
| `tests/runtime_attribute/` | A TOG reads the records of its work-item's key: a scalar DMA offset per key, a trip count and a block table (from the scalar or from the record's extent), a key that recorded 0 trips and issues nothing, and gathers whose requests are exactly the 32-byte blocks of the gathered addresses (a row gather with the trip as `at`, an element gather). Every loader check (no path, a path that does not open, no `programs.bin`, another ABI, keys dispatched and listed that differ, a header of another record, an undeclared id, another rank, another size, a negative extent, a stale key, a missing record, a foreign file) and every togsim.hpp misuse is refused by its message, exit 1, with `--tog_so` and in a session; a TOG without attributes ignores a path without records and refuses one with records; a declaration that breaks a rule does not compile, or is refused at load through `register_tog`. |
| `tests/semaphore/` | A barrier waits for every load of its semaphore; a broken contract is refused, with `--models_list` at the `DEVICE_SYNC` with a spad size and without. |
| `tests/session/` | `--models_list` with two kernels, on one partition and on two. A kernel launched behind another with no `DEVICE_SYNC` between is run whole. A kernel with an empty unit table may come first, and a later table may add a unit. |
| `tests/spad_capacity/` | A dispatch whose `.spad` is more than a lane's spad (`vpu_spad_size_kb_per_lane`) is refused while its TOG is loaded, with `--tog_so` and with `--models_list` (the first kernel or the second): one byte over, a later work-item, a negative one, one byte over tpuv6e's 1024 KB as shipped; one that fills the lane to the byte runs. Two work-items are in flight when twice the `.spad` fits and one at a time when it does not, by a byte. No instruction waits for spad: work-items moving more bytes than the lane trace as without a limit. No spad key refuses nothing; `core_spad_size_kb` is refused alone and may only restate the lanes. |
| `tests/typed/` | The deprecated `compute_type` 1, 2 and 3. What they pin is not typed-only: a semaphore re-wait, one compute per core, the array split. |
| `tests/unit_capacity/` | A compute tile whose unit table rows admit more than a unit takes in the tile's cycles is refused while its TOG is loaded, with `--tog_so` and with `--models_list` (the first kernel or the second): one more than the tile holds, for the array unit on one array, over two arrays and without the split, for a vector unit alone and beside a split array, and for a unit of capacity 12; the tile filled exactly runs. 128 rows on one array in 100 cycles are refused at five timings and statistics intervals, and a tile of 0 cycles admits nothing. A tile that no compute names is not measured. |
| `tests/unit/` | Tensor tiling and DMA addresses, without the simulator. |
| `tests/warnings/` | What still runs and is said once in a `[warning]` line: a `tile_id` without unit-table rows, an `arg_id` beyond `tog_tensors.txt`. |

`tests/examples/` does not check the numbers that the documents state in their other tables or in
prose; those were copied from the `expected.txt` files by hand.

## Adding a test

A test is a function in the `cases.py` of the directory it belongs to, or of a new directory, next
to the hand-written TOG (`*_tog.cpp`) it runs. `tests/lib/togtest.py` compiles the TOG, writes
the tables and the machine config, runs the simulator and reads its log. A new directory with one
case, running a copy of the hello TOG:

```bash
mkdir -p tests/mine
cp examples/hello/hello_tog.cpp tests/mine/my_tog.cpp
cat > tests/mine/cases.py <<'PY'
"""One line that says what these cases pin; tests/run.py --list prints it."""
from togtest import Case


def one_compute_runs_600_cycles(t):
    run = t.simulate(t.tog("my_tog.cpp", tensors=(16384, 16384)), t.machine(), cycles=[600],
                     units=[(0, "TPUV6E_MXU", "rows", 1, 128)])
    run.expect_total_cycles(840)


CASES = [Case("one_compute_runs_600_cycles", one_compute_runs_600_cycles)]
PY
python3 tests/run.py -k mine/
```

```text
PASS mine/one_compute_runs_600_cycles (0.3s)
ALL PASS (1 cases in 0.3s)
```

`t.machine()` is tpuv6e; `t.machine(num_cores=2)` changes a key. `run.computes()`, `run.events()`
and `run.expect_refused(...)` read the rest of the log; `tests/cycles/cases.py` and
`tests/semaphore/cases.py` are short models to copy from.
