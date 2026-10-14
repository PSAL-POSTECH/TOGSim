"""Every machine config under configs/ loads as it is shipped and replays a small TOG to its Total line."""
import yaml

from togtest import MIB, REPO, Case, Failure

CONFIGS = REPO / "configs"
SHIPPED = [
    "tpuv3_booksim",
    "tpuv3",
    "tpuv3_half",
    "tpuv4",
    "tpuv6e",
    "tpuv3_32x32",
    "tpuv3_8x8",
]


def units(name):
    """The unit table of the TOG on configs/<name>.yml: 256 rows of the array unit the config names
    (systolic_array_unit, SYSTOLIC by default), and 100 cycles of the Vpu."""
    array = yaml.safe_load((CONFIGS / f"{name}.yml").read_text()).get("systolic_array_unit", "SYSTOLIC")
    return [(0, array, "rows", 1, 256), (0, "Vpu", "cycles", 1, 100)]


def every_config_is_listed(t):
    """SHIPPED is configs/*.yml: a config added without a case, or one renamed or removed, fails here by name."""
    on_disk = sorted(p.stem for p in CONFIGS.glob("*.yml"))
    if on_disk != sorted(SHIPPED):
        raise Failure(f"expected configs/*.yml to be the {len(SHIPPED)} listed in tests/configs/cases.py; "
                      f"not listed: {sorted(set(on_disk) - set(SHIPPED))}, not in configs/: "
                      f"{sorted(set(SHIPPED) - set(on_disk))}")


def loads_and_runs(name):
    """configs/<name>.yml replays the TOG with a unit table to exit 0, a Total line, both computes finished and no
    key TOGSim does not read; without a unit table it is refused."""
    def case(t):
        tog, config = t.tog("load_compute_store_tog.cpp", tensors=(MIB, MIB)), CONFIGS / f"{name}.yml"
        run = t.simulate(tog, config, cycles=[600], units=units(name)).expect_ok()
        run.total_cycles()
        run.expect_no_warning("is not read by TOGSim")
        if len(run.computes()) != 2:
            run.fail(f"expected the computes of both work-items to finish, got {run.computes()}")
        t.simulate(tog, config, cycles=[600]).expect_refused("give --unit_table, or set systolic_array_split: none",
                                                             exit=1)
    return case


CASES = [Case("every_config_is_listed", every_config_is_listed)] + [
    Case(name, loads_and_runs(name)) for name in SHIPPED]
