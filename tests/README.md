# Tests

`tests/run.py` runs every case of every `tests/*/cases.py` against a built tree and prints one
`PASS` or `FAIL` line per case. It needs `g++`, `python3` (standard library only) and the default
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
| `tests/abi/` | A TOG built against another ABI version is refused; the same TOG at this version replays. |
| `tests/array_split/` | The array split of a segment TOG, and what it refuses. |
| `tests/configs/` | Every shipped config replays a small TOG, and is refused without a unit table. |
| `tests/cycle_table/` | A run without a cycle table, or with a `tile_id` the table does not hold, is refused. |
| `tests/cycles/` | `Total execution cycles` of four hand-written TOGs on tpuv6e, as goldens. |
| `tests/examples/` | `examples/run.sh <name>` prints `examples/<name>/expected.txt`; the fenced blocks of the READMEs and of `docs/` quote only lines of those files and of the examples' sources; every row of the README's refusal table is a refusal the mistakes example prints. |
| `tests/kernels/` | Each kernel of [`kernels/`](../docs/kernels.md) at its example shape: its `MOVIN`/`MOVOUT`/`COMP`/`MEMORY_BAR` counts and its `Total execution cycles`, as goldens; that gemm, conv and attention read their shape arguments; that conv, at a stride, a dilation and symmetric or asymmetric pads (a 1x1 at stride 2, a 7x7 at stride 2 with pad 3 over 3 channels among them), moves exactly the input, weight and output elements a convolution of that shape reads and writes, read from the address, size and stride of every DMA; each kernel at a shape none of its tiles divides (and conv short on every axis, and with 112 output columns in tiles of 32): no DMA ends past its tensor, every element is moved as often as the shape says, the computes name the tile ids of the short variants and no other, and the counts and the total are goldens; the shape arguments a kernel refuses (exit `1`). No run that checks instruction counts logs a warn-level line: the tables and `tog_tensors.txt` cover every tile id and `arg_id` the kernel names. |
| `tests/outputs/` | The shape of the lines a run logs, against golden files. |
| `tests/semaphore/` | A barrier waits for every load of its semaphore; a broken contract is refused, with `--models_list` and a spad size at the launch. |
| `tests/session/` | `--models_list` with two kernels, on one partition and on two. Two kernels at once on one core free only their own scratchpad bytes. |
| `tests/spad_capacity/` | A work-item that needs more of the spad at once than a core has (`core_spad_size_kb`) is refused while its TOG is loaded, with `--tog_so` and with `--models_list` (the first kernel or the second): one byte over, one KB short, a later work-item, a single load larger than the spad; a fill the reader does not wait for is not counted. One that fills the spad to the byte runs, and so does one that names more bytes than the spad and never holds them together, each to a pinned total; `core_spad_size_kb: 0`, or no such key, refuses nothing. A run that was measured traces what one that was not traces. |
| `tests/spad_footprint/` | Buffers are as large as the TOG states; an unsized one is refused. |
| `tests/typed/` | The deprecated `compute_type` 1, 2 and 3. What they pin is not typed-only: among it, the spad a work-item needs at once. |
| `tests/unit_capacity/` | A compute tile whose unit table rows admit more than a unit takes in the tile's cycles is refused while its TOG is loaded, with `--tog_so` and with `--models_list` (the first kernel or the second): one more than the tile holds, for the array unit on one array, over two arrays and without the split, for a vector unit alone and beside a split array, and for a unit of capacity 12; the tile filled exactly runs. 256 rows in 230 cycles are refused at five timings and statistics intervals, and a tile of 0 cycles admits nothing. A tile that no compute names is not measured. |
| `tests/unit/` | Tensor tiling and DMA addresses, without the simulator. |
| `tests/warnings/` | What still runs and is said once in a `[warning]` line: a `tile_id` without unit-table rows, an `arg_id` beyond `tog_tensors.txt`. |

`tests/examples/` does not check the numbers that the documents state in their other tables or in
prose; those were copied from the `expected.txt` files by hand.

## Adding a test

A test is a function in the `cases.py` of the directory it belongs to, or of a new directory, next
to the hand-written TOG (`*_tog.cpp`) it replays. `tests/lib/togtest.py` compiles the TOG, writes
the tables and the machine config, runs the simulator and reads its log. A new directory with one
case, replaying a copy of the hello TOG:

```bash
mkdir -p tests/mine
cp examples/hello/hello_tog.cpp tests/mine/my_tog.cpp
cat > tests/mine/cases.py <<'PY'
"""One line that says what these cases pin; tests/run.py --list prints it."""
from togtest import Case


def one_compute_runs_600_cycles(t):
    run = t.simulate(t.tog("my_tog.cpp", tensors=(16384, 16384)), t.machine(), cycles=[600],
                     units=[(0, "Systolic", "rows", 1, 256)])
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
