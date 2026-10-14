from __future__ import annotations

import yaml

from .tiling import Hardware

#: The keys a spad is stated by, and the most KB TOGSim holds one in (a uint32_t).
SPAD_KEYS = ("vpu_spad_size_kb_per_lane", "core_spad_size_kb")
SPAD_KB_MAX = (1 << 32) - 1


def core_spad_kb(cfg: dict) -> int:
    """The spad of one core in KB as TOGSim decides it (core_spad_kb, src/Common.cc): vpu_spad_size_kb_per_lane x
    vpu_num_lanes when the config gives both, and a core_spad_size_kb beside them must equal that product; without
    both, core_spad_size_kb alone; 0 when neither, no limit. A ValueError says what TOGSim would refuse."""
    stated = cfg.get("core_spad_size_kb")
    stated = None if stated is None else int(stated)
    if cfg.get("vpu_spad_size_kb_per_lane") is None or cfg.get("vpu_num_lanes") is None:
        return stated or 0
    kb_per_lane, lanes = int(cfg["vpu_spad_size_kb_per_lane"]), int(cfg["vpu_num_lanes"])
    kb = kb_per_lane * lanes
    if kb > SPAD_KB_MAX:
        raise ValueError(f"vpu_spad_size_kb_per_lane x vpu_num_lanes = {kb_per_lane} x {lanes} = {kb} KB is more "
                         f"than a spad TOGSim holds ({SPAD_KB_MAX} KB)")
    if stated is not None and stated != kb:
        raise ValueError(f"core_spad_size_kb: {stated} is not vpu_spad_size_kb_per_lane x vpu_num_lanes = "
                         f"{kb_per_lane} x {lanes} = {kb}: state the spad once, by the lanes or by core_spad_size_kb "
                         f"alone")
    return kb


def read_hardware(config_path: str, sa: int | None = None,
                  elem_bits: int = 32) -> Hardware:
    """The machine of the config at `config_path`, with `sa` or else vpu_num_lanes, the rows one array takes in
    TOGSim, and the spad TOGSim gives a core of it (core_spad_kb). A ValueError says what is missing, not an integer,
    or below what a machine has: one lane, one array row, one element per lane, no spad; or a spad TOGSim refuses."""
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{config_path}: not a mapping of config keys")

    try:
        vlane = int(cfg["vpu_num_lanes"])
        vlen = int(cfg["vpu_vector_length_bits"])
        spad = {key: int(cfg[key]) for key in SPAD_KEYS if cfg.get(key) is not None}
    except KeyError as exc:
        raise ValueError(f"{config_path}: missing required key {exc}") from None
    except TypeError:
        raise ValueError(f"{config_path}: vpu_num_lanes, vpu_vector_length_bits, vpu_spad_size_kb_per_lane and "
                         f"core_spad_size_kb are not all integers") from None
    resolved = sa if sa is not None else vlane
    for what, value, least in (("vpu_num_lanes", vlane, 1),
                               ("the systolic array dimension (--sa, else vpu_num_lanes)", resolved, 1),
                               ("vpu_vector_length_bits", vlen, elem_bits),
                               *((key, value, 0) for key, value in spad.items())):
        if value < least:
            raise ValueError(f"{config_path}: {what} is {value}, below {least}")
    try:
        spad_kb = core_spad_kb(cfg)
    except ValueError as exc:
        raise ValueError(f"{config_path}: {exc}") from None

    return Hardware(sa=resolved, vlane=vlane, vlen_bits=vlen, elem_bits=elem_bits, spad_bytes=spad_kb * 1024)
