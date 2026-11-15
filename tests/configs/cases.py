"""Every machine config under configs/ follows tpuv6e's convention, loads as it is shipped and replays a small TOG
to its Total line."""
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
    """The unit table of the TOG on configs/<name>.yml: 256 rows of the array unit named after its pytorchsim_* entry
    (TPUV6E_MXU for pytorchsim_tpuv6e), and 100 cycles of the Vpu."""
    isa = yaml.safe_load((CONFIGS / f"{name}.yml").read_text())["isa"]
    accel = next(str(entry) for entry in isa if str(entry).startswith("pytorchsim_"))
    return [(0, f"{accel[len('pytorchsim_'):].upper()}_MXU.rows", "rows", 1, 256), (0, "Vpu", "cycles", 1, 100)]


def splits(config):
    """Whether the config `config` spreads a compute's array rows over its arrays (systolic_array_split, round_robin
    by default): its cycle tables are then one array's, as gem5 measured them before vcix's tpu_mxu_units."""
    return config.get("systolic_array_split", "round_robin") != "none"


def every_config_is_listed(t):
    """SHIPPED is configs/*.yml: a config added without a case, or one renamed or removed, fails here by name."""
    on_disk = sorted(p.stem for p in CONFIGS.glob("*.yml"))
    if on_disk != sorted(SHIPPED):
        raise Failure(f"expected configs/*.yml to be the {len(SHIPPED)} listed in tests/configs/cases.py; "
                      f"not listed: {sorted(set(on_disk) - set(SHIPPED))}, not in configs/: "
                      f"{sorted(set(SHIPPED) - set(on_disk))}")


def breaks_convention(config):
    """What the config `config` (a parsed YAML mapping) lacks of the convention: an isa list with exactly one
    pytorchsim_* and one zvl<N>b entry, vcix and a non-empty address_map as mappings, and under the array split
    systolic_array_unit named after the pytorchsim_* entry as TPUV6E_MXU is after pytorchsim_tpuv6e; without it,
    no systolic_array_unit and vcix's tpu_mxu_units at num_systolic_array_per_core. Empty when it follows it."""
    isa = config.get("isa")
    accel = [entry for entry in isa if str(entry).startswith("pytorchsim_")] if isinstance(isa, list) else []
    lacks = [] if len(accel) == 1 else [f"an isa list with exactly one pytorchsim_* entry (isa: {isa})"]
    entries = [str(entry) for entry in isa] if isinstance(isa, list) else []
    zvl = [entry for entry in entries if entry[:3] == "zvl" and entry[-1:] == "b" and entry[3:-1].isdigit()]
    if isinstance(isa, list) and len(zvl) != 1:
        lacks.append(f"exactly one zvl<N>b entry in isa, the vector length (isa: {isa})")
    lacks += [key for key in ("vcix", "address_map") + (("systolic_array_unit",) if splits(config) else ())
              if key not in config]
    lacks += [f"{key} as a mapping ({key}: {config[key]!r})" for key in ("vcix", "address_map")
              if key in config and not isinstance(config[key], dict)]
    if config.get("address_map") == {}:
        lacks.append("address_map with its addresses (address_map: {})")
    unit = config.get("systolic_array_unit")
    named = accel[0][len("pytorchsim_"):].upper() + "_MXU" if len(accel) == 1 else None
    if "systolic_array_unit" in config and named and unit != named:
        lacks.append(f"systolic_array_unit named after {accel[0]} (systolic_array_unit: {unit!r})")
    if not splits(config):
        arrays = config.get("num_systolic_array_per_core", 1)
        if "systolic_array_unit" in config:
            lacks.append(f"no systolic_array_unit under systolic_array_split: none (systolic_array_unit: {unit!r})")
        vcix = config.get("vcix")
        if isinstance(vcix, dict) and vcix.get("tpu_mxu_units", 1) != arrays:
            lacks.append(f"vcix's tpu_mxu_units at num_systolic_array_per_core ({arrays}) under systolic_array_split: "
                         f"none, gem5 timing every array (tpu_mxu_units: {vcix.get('tpu_mxu_units')})")
    return lacks


def follows_convention(name):
    """configs/<name>.yml, read with PyYAML, states what breaks_convention asks for."""
    def case(t):
        lacks = breaks_convention(yaml.safe_load((CONFIGS / f"{name}.yml").read_text()))
        if lacks:
            raise Failure(f"expected configs/{name}.yml to follow tpuv6e's convention, it lacks: {'; '.join(lacks)}")
    return case


def convention_names_what_is_missing(t):
    """The check reads something: tpuv6e (and tpuv3, which splits) with each key taken away, or a second pytorchsim_*
    entry, is named."""
    v6e = yaml.safe_load((CONFIGS / "tpuv6e.yml").read_text())
    v3 = yaml.safe_load((CONFIGS / "tpuv3.yml").read_text())
    for name, config, keys in (("tpuv6e", v6e, ("isa", "vcix", "address_map")),
                               ("tpuv3", v3, ("isa", "vcix", "address_map", "systolic_array_unit"))):
        for key in keys:
            lacks = breaks_convention({k: v for k, v in config.items() if k != key})
            if len(lacks) != 1 or key not in lacks[0]:
                raise Failure(f"expected {name} without {key} to be named as lacking it alone, got {lacks}")
    for what, base, broken in (
            ("an isa with two pytorchsim_* entries", v6e, {"isa": v6e["isa"] + ["pytorchsim_tpuv4"]}),
            ("an isa without zvl<N>b", v6e, {"isa": [e for e in v6e["isa"] if not e.startswith("zvl")]}),
            ("vcix: null", v6e, {"vcix": None}), ("an empty address_map", v6e, {"address_map": {}}),
            ("systolic_array_unit: null", v3, {"systolic_array_unit": None}),
            ("TPUV6E_MXU beside pytorchsim_tpuv3", v3, {"systolic_array_unit": "TPUV6E_MXU"}),
            ("systolic_array_unit without the split", v6e, {"systolic_array_unit": "TPUV6E_MXU"}),
            ("no tpu_mxu_units without the split", v6e,
             {"vcix": {k: v for k, v in v6e["vcix"].items() if k != "tpu_mxu_units"}})):
        if not breaks_convention({**base, **broken}):
            raise Failure(f"expected {what} to break the convention, it did not")


def loads_and_runs(name):
    """configs/<name>.yml replays the TOG with a unit table to exit 0, a Total line, both computes finished and no
    key TOGSim does not read; without a unit table it is refused when it splits, and runs when it does not."""
    def case(t):
        tog, config = t.tog("load_compute_store_tog.cpp", tensors=(MIB, MIB)), CONFIGS / f"{name}.yml"
        run = t.simulate(tog, config, cycles=[600], units=units(name)).expect_ok()
        run.total_cycles()
        run.expect_no_warning("is not read by TOGSim")
        if len(run.computes()) != 2:
            run.fail(f"expected the computes of both work-items to finish, got {run.computes()}")
        bare = t.simulate(tog, config, cycles=[600])
        if splits(yaml.safe_load(config.read_text())):
            bare.expect_refused("give --unit_table, or set systolic_array_split: none", exit=1)
        else:
            bare.expect_ok().total_cycles()
    return case


CASES = [Case("every_config_is_listed", every_config_is_listed),
         Case("convention_names_what_is_missing", convention_names_what_is_missing)] + [
    Case(name, loads_and_runs(name)) for name in SHIPPED] + [
    Case(f"{name}.follows_convention", follows_convention(name)) for name in SHIPPED]
