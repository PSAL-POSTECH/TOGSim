"""dma_load_latency_cycles and dma_store_latency_cycles: the cycles the DMA engine holds a DMA before its first
request, so the DMAs pay them one after another. Each case runs a TOG with a key and without it, on the simple DRAM,
and states the difference in core cycles."""
import re
from pathlib import Path

from togtest import MIB, Case

TESTS = Path(__file__).resolve().parents[1]
WORK_ITEM = TESTS / "spad_capacity" / "work_item_tog.cpp"
BARRIER = TESTS / "semaphore" / "barrier_waits_both_loads_tog.cpp"
L = 1000
LOAD_READ = "L(1); C({1}, {})"
LOAD_COMPUTE_STORE = "L(1); C({1}, {2}); S(2)"
SIMPLE_DRAM = dict(dram_type="simple", dram_latency=100, ramulator_config_path=None)


def keys(load=None, store=None):
    return dict(dma_load_latency_cycles=load, dma_store_latency_cycles=store, **SIMPLE_DRAM)


def work_item(t, body, **machine):
    """One work-item of work_item_tog.cpp, the calls `body` over a 16 KB load and a 16 KB result, compute 200."""
    tog = t.tog(WORK_ITEM, (f"BODY={body}", "SIZES=0,16384,16384", "ITEMS=1"), tensors=(256 * MIB, 256 * MIB))
    return t.simulate(tog, t.machine(systolic_array_split="none", **machine), cycles=[200]).expect_ok()


def barrier(t, **machine):
    """The semaphore TOG: two async loads, one barrier on both, one compute."""
    return t.simulate(t.tog(BARRIER), t.machine(systolic_array_split="none", **machine), cycles=[200]).expect_ok()


def grows_by(run, base, cycles):
    if run.total_cycles() - base.total_cycles() != cycles:
        run.fail(f"expected the total to grow by {cycles}: {base.total_cycles()} -> {run.total_cycles()}")


def zero_is_absent(t):
    """Both keys at 0 give the trace lines and total of no key. Breaks if a latency of 0 still holds the engine."""
    for replay in (lambda **k: work_item(t, LOAD_COMPUTE_STORE, **k), lambda **k: barrier(t, **k)):
        base, zero = replay(**SIMPLE_DRAM), replay(**keys(0, 0))
        seen = lambda run: [(e.cycle, e.tag, e.inst) for e in run.events()]
        if seen(zero) != seen(base) or zero.total_cycles() != base.total_cycles():
            zero.fail(f"expected the keys at 0 to change nothing: total {base.total_cycles()} -> {zero.total_cycles()}")


def dma_cycles(run):
    """(active, idle) of the last `DMA active_cycles` line, the run's totals."""
    return tuple(map(int, re.findall(r"DMA active_cycles: (\d+), DMA idle_cycles: (\d+)", run.log)[-1]))


def one_load(t):
    """A load and its reader: the load and the run finish exactly L later, and the hold counts as DMA idle, not
    active."""
    base, run = work_item(t, LOAD_READ, **SIMPLE_DRAM), work_item(t, LOAD_READ, **keys(load=L))
    grows_by(run, base, L)
    (active, idle), (base_active, base_idle) = dma_cycles(run), dma_cycles(base)
    if (active, idle) != (base_active, base_idle + L):
        run.fail(f"expected DMA active {base_active} and idle {base_idle + L}, got {active} and {idle}")
    if run.when("INST_FINISHED", 0) != base.when("INST_FINISHED", 0) + L:
        run.fail(f"expected the load to finish @{base.when('INST_FINISHED', 0) + L}, "
                 f"got @{run.when('INST_FINISHED', 0)}")


def async_loads_take_turns(t):
    """Two async loads on one barrier: the first sends its requests L later, the second 2L later, and the run grows by
    2L. Breaks if the holds of DMAs in flight overlap."""
    base, run = barrier(t, **SIMPLE_DRAM), barrier(t, **keys(load=L))
    grows_by(run, base, 2 * L)
    for inst, k in ((0, 1), (1, 2)):
        if run.when("ASYNC_DMA_ISSUE", inst) != base.when("ASYNC_DMA_ISSUE", inst) + k * L:
            run.fail(f"expected load {inst}'s ASYNC_DMA_ISSUE {k * L} after @{base.when('ASYNC_DMA_ISSUE', inst)}, "
                     f"got @{run.when('ASYNC_DMA_ISSUE', inst)}")


def store_takes_its_turn(t):
    """A load, a compute and a store: only the store key set, the store and the run finish L later."""
    base, run = work_item(t, LOAD_COMPUTE_STORE, **SIMPLE_DRAM), work_item(t, LOAD_COMPUTE_STORE, **keys(store=L))
    grows_by(run, base, L)
    if (run.when("INST_FINISHED", 0), run.when("INST_FINISHED", 2)) != (base.when("INST_FINISHED", 0),
                                                                         base.when("INST_FINISHED", 2) + L):
        run.fail("expected the load to stay and the store to finish L later")


def latency_longer_than_the_wedge_guard(t):
    """A hold of 150000 cycles, past the wedge guard's 100000 frozen cycles: the run is not declared wedged."""
    grows_by(work_item(t, LOAD_READ, **keys(load=150000)), work_item(t, LOAD_READ, **SIMPLE_DRAM), 150000)


CASES = [
    Case("zero_is_absent", zero_is_absent),
    Case("one_load", one_load),
    Case("async_loads_take_turns", async_loads_take_turns),
    Case("store_takes_its_turn", store_takes_its_turn),
    Case("latency_longer_than_the_wedge_guard", latency_longer_than_the_wedge_guard),
]
