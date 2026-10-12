from __future__ import annotations

import os
import re

import yaml

from .tiling import Hardware

#: systolic_ws_<SA>x<SA>_... -- the only place SA is written down.
_SA_IN_NAME = re.compile(r"_(\d+)x(\d+)_")


def array_dim_from_name(config_path: str) -> int | None:
    m = _SA_IN_NAME.search(os.path.basename(config_path))
    if not m or m.group(1) != m.group(2):
        return None
    return int(m.group(1))


def read_hardware(config_path: str, sa: int | None = None,
                  elem_bits: int = 32) -> Hardware:
    """The machine of the config at `config_path`, with `sa` or else the array dimension its name carries. A
    ValueError says what is missing, not an integer, or below what a machine has: one array row, one lane, one
    element per lane, no spad."""
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{config_path}: not a mapping of config keys")

    resolved = sa if sa is not None else array_dim_from_name(config_path)
    if resolved is None:
        raise ValueError(
            f"cannot tell the systolic array dimension from "
            f"{os.path.basename(config_path)!r}: it is not a TOGSim config key and "
            f"the name does not carry it. Pass --sa.")

    try:
        vlane = int(cfg["vpu_num_lanes"])
        vlen = int(cfg["vpu_vector_length_bits"])
        spad_kb = int(cfg.get("core_spad_size_kb") or 0)
    except KeyError as exc:
        raise ValueError(f"{config_path}: missing required key {exc}") from None
    except TypeError:
        raise ValueError(f"{config_path}: vpu_num_lanes, vpu_vector_length_bits and core_spad_size_kb are not all "
                         f"integers") from None
    for what, value, least in (("the systolic array dimension (--sa, else the config's name)", resolved, 1),
                               ("vpu_num_lanes", vlane, 1), ("vpu_vector_length_bits", vlen, elem_bits),
                               ("core_spad_size_kb", spad_kb, 0)):
        if value < least:
            raise ValueError(f"{config_path}: {what} is {value}, below {least}")

    return Hardware(sa=resolved, vlane=vlane, vlen_bits=vlen, elem_bits=elem_bits, spad_bytes=spad_kb * 1024)
