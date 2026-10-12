"""TensorTiling and the DRAM addresses of a MOVIN, checked without the simulator by build/bin/tensor_tiling_test."""
from togtest import Case

CHECKS = ["row_major_is_pitched_runs", "tiled_is_one_run", "footprint_pads_to_whole_tiles"]


def check(name):
    """The named check of tests/unit/tensor_tiling.cc prints its `ok` line and exits 0."""
    def case(t):
        run = t.execute("tensor_tiling_test", [name])
        if run.rc != 0 or not run.log.startswith("ok "):
            run.fail(f"expected tensor_tiling_test {name} to print ok and exit 0, got exit {run.rc}: "
                     f"{run.log.strip() or 'no output'}")
    return case


CASES = [Case(f"tensor_tiling.{name}", check(name)) for name in CHECKS]
