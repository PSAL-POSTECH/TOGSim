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
    """The unit table of the TOG on configs/<name>.yml: 256 rows of the array unit the config names
    (systolic_array_unit), and 100 cycles of the Vpu."""
    array = yaml.safe_load((CONFIGS / f"{name}.yml").read_text())["systolic_array_unit"]
    return [(0, array, "rows", 1, 256), (0, "Vpu", "cycles", 1, 100)]


def every_config_is_listed(t):
    """SHIPPED is configs/*.yml: a config added without a case, or one renamed or removed, fails here by name."""
    on_disk = sorted(p.stem for p in CONFIGS.glob("*.yml"))
    if on_disk != sorted(SHIPPED):
        raise Failure(f"expected configs/*.yml to be the {len(SHIPPED)} listed in tests/configs/cases.py; "
                      f"not listed: {sorted(set(on_disk) - set(SHIPPED))}, not in configs/: "
                      f"{sorted(set(SHIPPED) - set(on_disk))}")


def breaks_convention(config):
    """What the config `config` (a parsed YAML mapping) lacks of the convention: an isa list with exactly one
    pytorchsim_* and one zvl<N>b entry, vcix and a non-empty address_map as mappings, and systolic_array_unit
    named after the pytorchsim_* entry as TPUV6E_MXU is after pytorchsim_tpuv6e. Empty when it follows it."""
    isa = config.get("isa")
    accel = [entry for entry in isa if str(entry).startswith("pytorchsim_")] if isinstance(isa, list) else []
    lacks = [] if len(accel) == 1 else [f"an isa list with exactly one pytorchsim_* entry (isa: {isa})"]
    entries = [str(entry) for entry in isa] if isinstance(isa, list) else []
    zvl = [entry for entry in entries if entry[:3] == "zvl" and entry[-1:] == "b" and entry[3:-1].isdigit()]
    if isinstance(isa, list) and len(zvl) != 1:
        lacks.append(f"exactly one zvl<N>b entry in isa, the vector length (isa: {isa})")
    lacks += [key for key in ("vcix", "address_map", "systolic_array_unit") if key not in config]
    lacks += [f"{key} as a mapping ({key}: {config[key]!r})" for key in ("vcix", "address_map")
              if key in config and not isinstance(config[key], dict)]
    if config.get("address_map") == {}:
        lacks.append("address_map with its addresses (address_map: {})")
    unit = config.get("systolic_array_unit")
    if "systolic_array_unit" in config and len(accel) == 1 and unit != accel[0][len("pytorchsim_"):].upper() + "_MXU":
        lacks.append(f"systolic_array_unit named after {accel[0]} (systolic_array_unit: {unit!r})")
    return lacks


def follows_convention(name):
    """configs/<name>.yml, read with PyYAML, states what breaks_convention asks for."""
    def case(t):
        lacks = breaks_convention(yaml.safe_load((CONFIGS / f"{name}.yml").read_text()))
        if lacks:
            raise Failure(f"expected configs/{name}.yml to follow tpuv6e's convention, it lacks: {'; '.join(lacks)}")
    return case


def convention_names_what_is_missing(t):
    """The check reads something: tpuv6e with each key taken away, or a second pytorchsim_* entry, is named."""
    v6e = yaml.safe_load((CONFIGS / "tpuv6e.yml").read_text())
    for key in ("isa", "vcix", "address_map", "systolic_array_unit"):
        lacks = breaks_convention({k: v for k, v in v6e.items() if k != key})
        if len(lacks) != 1 or key not in lacks[0]:
            raise Failure(f"expected tpuv6e without {key} to be named as lacking it alone, got {lacks}")
    for what, broken in (("an isa with two pytorchsim_* entries", {"isa": v6e["isa"] + ["pytorchsim_tpuv4"]}),
                         ("an isa without zvl<N>b", {"isa": [e for e in v6e["isa"] if not e.startswith("zvl")]}),
                         ("vcix: null", {"vcix": None}), ("an empty address_map", {"address_map": {}}),
                         ("systolic_array_unit: null", {"systolic_array_unit": None}),
                         ("TPUV3_MXU beside pytorchsim_tpuv6e", {"systolic_array_unit": "TPUV3_MXU"})):
        if not breaks_convention({**v6e, **broken}):
            raise Failure(f"expected {what} to break the convention, it did not")


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


CASES = [Case("every_config_is_listed", every_config_is_listed),
         Case("convention_names_what_is_missing", convention_names_what_is_missing)] + [
    Case(name, loads_and_runs(name)) for name in SHIPPED] + [
    Case(f"{name}.follows_convention", follows_convention(name)) for name in SHIPPED]
