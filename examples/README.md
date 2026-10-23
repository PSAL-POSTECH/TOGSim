# Examples

Hand-written TOGs that the [tutorial](../docs/tutorial.md) walks through, one step
each, written against `include/togsim.hpp` (TOGSim API v1.0.0). Build TOGSim first, then from the
repository root:

```bash
examples/run.sh            # every example
examples/run.sh hello      # one of them
```

`run.sh` echoes each command it runs after `+` and prints the result lines of each log without
their timestamp. `<name>/expected.txt` is that output, and `tests/examples/` checks it on every
test run.

| Directory | What it shows | `Total execution cycles` of its first run |
|---|---|---|
| `hello/` | The smallest TOG: one load, one compute, one store, and the three files a run needs. Its last run prints a whole log. | 2437 |
| `async_barrier/` | Two asynchronous loads and the barrier that waits for them. `-DSYNC` builds the synchronous form, `-DTWO_STEPS` a variant where the two forms differ. | 2479 |
| `tiles/` | A kernel that dispatches a grid of work-items from `tog_shape.txt`: on one, two and four cores, in a small scratchpad, and with half the DRAM channels. | 8301 |
| `gemm/` | A GEMM-like work-item: two kinds of compute, an accumulator, and how the unit table's rows and the array split change a compute's cycles. | 2719 |
| `session/` | Two kernels launched from a command stream (`--models_list`), on one partition and on two. | 4916 |
| `mistakes/` | What TOGSim and togsim.hpp refuse, with the message of each of 18 refusals; two mistakes TOGSim only warns about; a consumer called before its producer. | 2437 |
| `minimal.yml` | A machine config that needs no other file and no unit table. | |

To add an example, add a directory with its sources and inputs, a function of the directory's name
in `run.sh`, and that name to the `EXAMPLES` list at the top of `run.sh`; then pin its output with
`examples/run.sh <name> > examples/<name>/expected.txt`.

Building leaves `*.so` and `*.gen.*` files beside the sources; they are ignored by git.
