"""Total execution cycles of fixed hand-written TOGs on tpuv6e: the number TOGSim exists to report, as goldens.
A golden moves only with a deliberate change to the timing model, a shipped config or a submodule; say which."""
from pathlib import Path

from togtest import Case

TESTS = Path(__file__).resolve().parents[1]
GOLDEN = {
    "two_async_loads_and_a_barrier": dict(
        total=892, tog="semaphore/barrier_waits_both_loads_tog.cpp", cycles=[200],
        machine=dict(systolic_array_split="none")),
    "piecewise_loads_and_stores_in_a_48kb_spad": dict(
        total=1536, tog="spad_footprint/piecewise_fill_tog.cpp", cycles=[200],
        machine=dict(core_spad_size_kb=48, systolic_array_split="none")),
    "indirect_store_without_its_index_dump": dict(
        total=649, tog="spad_footprint/indirect_store_tog.cpp", cycles=[200],
        machine=dict(core_spad_size_kb=48, systolic_array_split="none")),
    "computes_split_over_two_arrays": dict(
        total=3121, tog="spad_footprint/state_stored_each_trip_tog.cpp", cycles=[600], compute=344,
        units=[(0, "Systolic", "rows", 1, 512), (0, "Vpu", "cycles", 1, 50)], machine=dict()),
}


def golden(total, tog, cycles, machine, units=None, compute=None):
    """The TOG (a segment TOG of another directory) replays to `total` cycles; with `compute`, every compute states
    that many cycles (600 - 512 + 512/2 under the default systolic_array_split, for a compute_type 0 tile)."""
    def case(t):
        run = t.simulate(t.tog(TESTS / tog), t.machine(**machine), cycles, units)
        run.expect_total_cycles(total)
        seen = sorted({c.cycles for c in run.computes()})
        if compute is not None and seen != [compute]:
            run.fail(f"expected every compute to run {compute} cycles, got {seen}")
    return case


CASES = [Case(name, golden(**how)) for name, how in GOLDEN.items()]
