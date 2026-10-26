"""Total execution cycles of fixed hand-written TOGs on tpuv6e: the number TOGSim exists to report, as goldens.
A golden moves only with a deliberate change to the timing model, a shipped config or a submodule; say which."""
from pathlib import Path

from togtest import Case

TESTS = Path(__file__).resolve().parents[1]
GOLDEN = {
    "two_async_loads_and_a_barrier": dict(
        total=901, tog="semaphore/barrier_waits_both_loads_tog.cpp", cycles=[200],
        machine=dict(systolic_array_split="none")),
    "piecewise_loads_and_stores": dict(
        total=1067, tog="cycles/piecewise_fill_tog.cpp", cycles=[200], machine=dict(systolic_array_split="none")),
    "indirect_store_with_offsets_0": dict(
        total=647, tog="cycles/indirect_store_tog.cpp", cycles=[200],
        machine=dict(systolic_array_split="none"), records={(0, 0): [0] * 64, (1, 0): [0] * 64}),
    "computes_split_over_two_arrays": dict(
        total=2993, tog="cycles/state_stored_each_trip_tog.cpp", cycles=[600], compute=344,
        units=[(0, "TPUV6E_MXU.rows", "rows", 1, 512), (0, "Vpu", "cycles", 1, 50)], machine=dict()),
}


def golden(total, tog, cycles, machine, units=None, compute=None, records=None):
    """The TOG (a segment TOG of this or another directory) replays to `total` cycles; with `compute`, every compute states
    that many cycles (600 - 512 + 512/2 under the default systolic_array_split, for a compute_type 0 tile); `records`
    are its runtime attributes."""
    def case(t):
        run = t.simulate(t.tog(TESTS / tog), t.machine(**machine), cycles, units,
                         attributes=None if records is None else t.attributes(records))
        run.expect_total_cycles(total)
        seen = sorted({c.cycles for c in run.computes()})
        if compute is not None and seen != [compute]:
            run.fail(f"expected every compute to run {compute} cycles, got {seen}")
    return case


CASES = [Case(name, golden(**how)) for name, how in GOLDEN.items()]
