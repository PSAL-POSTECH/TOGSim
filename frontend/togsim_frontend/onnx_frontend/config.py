from __future__ import annotations

import yaml

from .tiling import Hardware

#: The keys a spad is stated by.
SPAD_KEYS = ("vpu_spad_size_kb_per_lane", "core_spad_size_kb")


def check_spad(cfg: dict) -> None:
    """A ValueError for a spad TOGSim refuses (spad_bytes_per_lane, src/Common.cc): core_spad_size_kb without
    vpu_spad_size_kb_per_lane, or beside it other than vpu_spad_size_kb_per_lane x vpu_num_lanes. TOGSim measures
    each dispatch's .spad against the lane's spad itself, while it loads the TOG."""
    stated = cfg.get("core_spad_size_kb")
    if stated is None:
        return
    if cfg.get("vpu_spad_size_kb_per_lane") is None:
        raise ValueError("core_spad_size_kb: a work-item states its spad per lane, so the spad is stated by "
                         "vpu_spad_size_kb_per_lane (and vpu_num_lanes)")
    kb_per_lane, lanes = int(cfg["vpu_spad_size_kb_per_lane"]), int(cfg.get("vpu_num_lanes") or 0)
    if int(stated) != kb_per_lane * lanes:
        raise ValueError(f"core_spad_size_kb: {int(stated)} is not vpu_spad_size_kb_per_lane x vpu_num_lanes = "
                         f"{kb_per_lane} x {lanes} = {kb_per_lane * lanes}: state the spad once, by the lanes")


def vector_length(cfg: dict) -> int:
    """The vector length in bits, from the zvl<N>b entry of the config's isa (RISC-V Zvl<N>b: VLEN >= N). A ValueError
    says the config has no isa, or an isa without exactly one zvl<N>b entry."""
    if "isa" not in cfg:
        raise ValueError("missing required key 'isa'")
    isa = cfg["isa"]
    if not isinstance(isa, list):
        raise ValueError(f"isa: {isa!r} is not a list of extensions")
    zvl = [e for e in isa if isinstance(e, str) and e.startswith("zvl") and e.endswith("b") and e[3:-1].isdigit()]
    if len(zvl) != 1:
        raise ValueError(f"isa: {isa} has {len(zvl) or 'no'} zvl<N>b entries, and the frontend takes the vector "
                         f"length from exactly one")
    return int(zvl[0][3:-1])


def read_hardware(config_path: str, sa: int | None = None,
                  elem_bits: int = 32) -> Hardware:
    """The machine of the config at `config_path`: `sa` or else vpu_num_lanes, the vector length (vector_length), the
    array unit (systolic_array_unit; None with systolic_array_split none). A ValueError says what is missing, not an
    integer, below one lane, array row or element per lane, or a spad TOGSim refuses (check_spad)."""
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{config_path}: not a mapping of config keys")

    try:
        vlane = int(cfg["vpu_num_lanes"])
        spad = {key: int(cfg[key]) for key in SPAD_KEYS if cfg.get(key) is not None}
    except KeyError as exc:
        raise ValueError(f"{config_path}: missing required key {exc}") from None
    except TypeError:
        raise ValueError(f"{config_path}: vpu_num_lanes, vpu_spad_size_kb_per_lane and core_spad_size_kb are not all "
                         f"integers") from None
    try:
        vlen = vector_length(cfg)
    except ValueError as exc:
        raise ValueError(f"{config_path}: {exc}") from None
    resolved = sa if sa is not None else vlane
    for what, value, least in (("vpu_num_lanes", vlane, 1),
                               ("the systolic array dimension (--sa, else vpu_num_lanes)", resolved, 1),
                               ("the vector length (zvl<N>b of isa)", vlen, elem_bits),
                               *((key, value, 0) for key, value in spad.items())):
        if value < least:
            raise ValueError(f"{config_path}: {what} is {value}, below {least}")
    try:
        check_spad(cfg)
    except ValueError as exc:
        raise ValueError(f"{config_path}: {exc}") from None
    split = cfg.get("systolic_array_split", "round_robin")
    if split not in ("none", "round_robin"):
        raise ValueError(f"{config_path}: systolic_array_split: {split} (none | round_robin)")
    unit = cfg.get("systolic_array_unit")
    if "systolic_array_unit" in cfg and not (isinstance(unit, str) and unit):
        raise ValueError(f"{config_path}: systolic_array_unit: {unit!r} is not a name, the unit table's name of the "
                         f"array unit")
    if unit is None and split == "round_robin":
        raise ValueError(f"{config_path}: systolic_array_split: round_robin (the default) needs systolic_array_unit, "
                         f"the unit table's name of the array unit")
    if unit is not None and split == "none":
        raise ValueError(f"{config_path}: systolic_array_unit: {unit} is ignored without systolic_array_split: "
                         f"round_robin")

    return Hardware(sa=resolved, vlane=vlane, vlen_bits=vlen, array_unit=unit, elem_bits=elem_bits)
