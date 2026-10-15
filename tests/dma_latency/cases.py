"""dma_load_latency_cycles and dma_store_latency_cycles: the cycles the DMA engine holds a DMA before its first
request, so the DMAs pay them one after another. dma_load_ready_cycles and dma_store_ready_cycles: the cycles a DMA
waits after its last data, not holding the engine, so DMAs in flight together wait at once. Each case runs a TOG with
a key and without it, on the simple DRAM, and states the difference in core cycles."""
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


def one_load(t):
    """A load and its reader: the load and the run finish exactly L later."""
    base, run = work_item(t, LOAD_READ, **SIMPLE_DRAM), work_item(t, LOAD_READ, **keys(load=L))
    grows_by(run, base, L)
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


def ready(load=None, store=None):
    return dict(dma_load_ready_cycles=load, dma_store_ready_cycles=store, **SIMPLE_DRAM)


def ready_zero_is_absent(t):
    """Both ready keys at 0 give the trace lines and total of no key."""
    for replay in (lambda **k: work_item(t, LOAD_COMPUTE_STORE, **k), lambda **k: barrier(t, **k)):
        base, zero = replay(**SIMPLE_DRAM), replay(**ready(0, 0))
        seen = lambda run: [(e.cycle, e.tag, e.inst) for e in run.events()]
        if seen(zero) != seen(base) or zero.total_cycles() != base.total_cycles():
            zero.fail(f"expected the keys at 0 to change nothing: total {base.total_cycles()} -> {zero.total_cycles()}")


def one_load_ready(t):
    """A load and its reader: the load and the run finish exactly L later."""
    base, run = work_item(t, LOAD_READ, **SIMPLE_DRAM), work_item(t, LOAD_READ, **ready(load=L))
    grows_by(run, base, L)
    if run.when("INST_FINISHED", 0) != base.when("INST_FINISHED", 0) + L:
        run.fail(f"expected the load to finish @{base.when('INST_FINISHED', 0) + L}, "
                 f"got @{run.when('INST_FINISHED', 0)}")


def async_loads_wait_together(t):
    """Two async loads on one barrier: both send their requests as before, and the run grows by L, not 2L. Breaks if
    a load's ready latency holds the engine."""
    base, run = barrier(t, **SIMPLE_DRAM), barrier(t, **ready(load=L))
    grows_by(run, base, L)
    for inst in (0, 1):
        if run.when("ASYNC_DMA_ISSUE", inst) != base.when("ASYNC_DMA_ISSUE", inst):
            run.fail(f"expected load {inst}'s ASYNC_DMA_ISSUE to stay @{base.when('ASYNC_DMA_ISSUE', inst)}, "
                     f"got @{run.when('ASYNC_DMA_ISSUE', inst)}")


def store_ready_waits_at_the_end(t):
    """A load, a compute and a store: only the store ready key set, the store finishes as before and the run
    L after it."""
    base = work_item(t, LOAD_COMPUTE_STORE, **SIMPLE_DRAM)
    run = work_item(t, LOAD_COMPUTE_STORE, **ready(store=L))
    if run.when("INST_FINISHED", 2) != base.when("INST_FINISHED", 2):
        run.fail(f"expected the store to finish @{base.when('INST_FINISHED', 2)}, got @{run.when('INST_FINISHED', 2)}")
    if run.total_cycles() < run.when("INST_FINISHED", 2) + L:
        run.fail(f"expected the run to end at least L after the store: store @{run.when('INST_FINISHED', 2)}, "
                 f"total {run.total_cycles()}")


CASES = [
    Case("zero_is_absent", zero_is_absent),
    Case("one_load", one_load),
    Case("async_loads_take_turns", async_loads_take_turns),
    Case("store_takes_its_turn", store_takes_its_turn),
    Case("latency_longer_than_the_wedge_guard", latency_longer_than_the_wedge_guard),
    Case("ready_zero_is_absent", ready_zero_is_absent),
    Case("one_load_ready", one_load_ready),
    Case("async_loads_wait_together", async_loads_wait_together),
    Case("store_ready_waits_at_the_end", store_ready_waits_at_the_end),
]
