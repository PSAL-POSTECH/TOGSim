"""systolic_array_split with a segment TOG: a compute's TPUV6E_MXU rows are spread over the arrays of its core, and
what the split refuses, most of it before any TOG is loaded: a unit table or a config it cannot read."""
from pathlib import Path

from togtest import MIB, Case

TOG = Path(__file__).resolve().parents[1] / "spad_footprint" / "state_stored_each_trip_tog.cpp"
UNITS = [(0, "TPUV6E_MXU", "rows", 1, 512), (0, "Vpu", "cycles", 1, 50)]
REFUSED = {
    "refused.unit_table_without_array_unit": dict(
        saying="systolic_array_split: the unit table has no array unit TPUV6E_MXU", exit=1, units=UNITS[1:]),
    "refused.array_unit_named_without_split": dict(
        saying="systolic_array_unit: MSA is ignored without systolic_array_split: round_robin", exit=134,
        systolic_array_split="none", systolic_array_unit="MSA"),
    "refused.split_without_array_unit": dict(
        saying="systolic_array_split: round_robin (the default) needs systolic_array_unit", exit=134,
        systolic_array_unit=None),
    "refused.array_unit_empty": dict(
        saying='systolic_array_unit: "" is not a name', exit=134, systolic_array_unit='""'),
    "refused.array_unit_null": dict(
        saying="systolic_array_unit: ~ is not a name", exit=134, systolic_array_unit="~"),
    "refused.split_without_unit_table": dict(
        saying="give --unit_table, or set systolic_array_split: none", exit=1, units=None),
    "refused.spread_exceeds_one_array": dict(
        saying="computes tile id 0, whose unit table row admits 128 rows to TPUV6E_MXU, and the unit takes 100 in the "
               "tile's 100 cycles (1 per cycle)", exit=1, num_systolic_array_per_core=4, cycles=100,
        units=[(0, "TPUV6E_MXU", "rows", 1, 128), (0, "Vpu", "cycles", 1, 50)]),
}


def replay(t, cycles=600, units=UNITS, **machine):
    """A segment TOG of tests/spad_footprint/ (compute_type 0, two work-items of four computes) on tpuv6e (2 arrays
    per core, 128 lanes) with `machine` set: its one tile runs `cycles`, `units` is the unit table."""
    return t.simulate(t.tog(TOG, tensors=(MIB, MIB)), t.machine(**machine), [cycles], units)


def computes_split_over_two_arrays(t):
    """The control of the refusals: every compute states 344 = 600 - 512 + 512/2 cycles (Core::split_compute_cycles)."""
    run = replay(t).expect_ok()
    seen = [c.cycles for c in run.computes()]
    if seen != [344] * 8:
        run.fail(f"expected the eight computes to run 344 cycles each (600 - 512 + 512/2), got {seen}")


def refused(saying, exit, **given):
    """The run is refused with `saying` and that exit status: 1, or 134 when the config is rejected (an abort)."""
    return lambda t: replay(t, **given).expect_refused(saying, exit)


CASES = [Case("computes_split_over_two_arrays", computes_split_over_two_arrays)] + [
    Case(name, refused(**how)) for name, how in REFUSED.items()]
