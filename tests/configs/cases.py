"""Every machine config under configs/ loads as it is shipped and replays a small TOG to its Total line."""
from togtest import MIB, REPO, Case, Failure

CONFIGS = REPO / "configs"
SHIPPED = [
    "systolic_ws_128x128_c1_booksim_tpuv2",
    "systolic_ws_128x128_c1_booksim_tpuv3",
    "systolic_ws_128x128_c1_simple_noc_tpuv2",
    "systolic_ws_128x128_c1_simple_noc_tpuv3",
    "systolic_ws_128x128_c1_simple_noc_tpuv3_functional_only",
    "systolic_ws_128x128_c1_simple_noc_tpuv3_half",
    "systolic_ws_128x128_c1_simple_noc_tpuv3_timing_only",
    "systolic_ws_128x128_c1_simple_noc_tpuv4",
    "systolic_ws_128x128_c2_booksim_tpuv3",
    "systolic_ws_128x128_c2_simple_noc_tpuv2",
    "systolic_ws_128x128_c2_simple_noc_tpuv3",
    "systolic_ws_128x128_c2_simple_noc_tpuv3_ils",
    "systolic_ws_128x128_c2_simple_noc_tpuv3_partition",
    "systolic_ws_128x128_c2_simple_noc_tpuv4",
    "systolic_ws_256x256_c1_simple_noc_tpuv6e",
    "systolic_ws_256x256_c1_simple_noc_tpuv6e_functional_only",
    "systolic_ws_256x256_c1_simple_noc_tpuv6e_timing_only",
    "systolic_ws_32x32_c1_simple_noc_tpuv3",
    "systolic_ws_8x8_c1_simple_noc_tpuv3",
]
RUNS_WITHOUT_UNIT_TABLE = []
UNITS = [(0, "Systolic", "rows", 1, 256), (0, "Vpu", "cycles", 1, 100)]


def every_config_is_listed(t):
    """SHIPPED is configs/*.yml: a config added without a case, or one renamed or removed, fails here by name."""
    on_disk = sorted(p.stem for p in CONFIGS.glob("*.yml"))
    if on_disk != sorted(SHIPPED):
        raise Failure(f"expected configs/*.yml to be the {len(SHIPPED)} listed in tests/configs/cases.py; "
                      f"not listed: {sorted(set(on_disk) - set(SHIPPED))}, not in configs/: "
                      f"{sorted(set(SHIPPED) - set(on_disk))}")


def loads_and_runs(name):
    """configs/<name>.yml, read where it is, replays the TOG with a unit table to exit 0, a Total line and both
    computes finished. Without a unit table it is refused, unless it is one of RUNS_WITHOUT_UNIT_TABLE (a config
    that sets systolic_array_split: none), which must then run to the end as well."""
    def case(t):
        tog, config = t.tog("load_compute_store_tog.cpp", tensors=(MIB, MIB)), CONFIGS / f"{name}.yml"
        run = t.simulate(tog, config, cycles=[600], units=UNITS).expect_ok()
        run.total_cycles()
        if len(run.computes()) != 2:
            run.fail(f"expected the computes of both work-items to finish, got {run.computes()}")
        bare = t.simulate(tog, config, cycles=[600])
        if name in RUNS_WITHOUT_UNIT_TABLE:
            bare.expect_ok().total_cycles()
        else:
            bare.expect_refused("give --unit_table, or set systolic_array_split: none", exit=1)
    return case


CASES = [Case("every_config_is_listed", every_config_is_listed)] + [
    Case(name, loads_and_runs(name)) for name in SHIPPED]
