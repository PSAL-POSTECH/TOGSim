"""
Generate the Ramulator 2.1 configs TOGSim ships, one per TPU memory, with the
DSL in extern/ramulator2/python.

Usage:
    python gen_configs.py

A TPU's memory is a named organization and timing preset below; Ramulator
resolves the rest of the timings from them.
"""

import json
import sys
import os

# Add ramulator2 Python DSL to path
RAMULATOR_PYTHON = os.path.join(os.path.dirname(__file__),
                                "../../extern/ramulator2/python")
sys.path.insert(0, RAMULATOR_PYTHON)

import ramulator
import ramulator.dram
import ramulator.controller
import ramulator.scheduler
import ramulator.refresh_manager
import ramulator.row_policy
import ramulator.addr_mapper
import ramulator.channel_mapper
import ramulator.memory_system


# TPUv3 and TPUv4: 1 KB rows (HBM's column address already counts the burst); the densities give
# the refresh times TOGSim has used, tRFC 350 ns and tRFCpb 160 ns.
ramulator.dram.HBM2.org_presets["HBM2_TPU"] = {
    "pseudochannel_density": 4096, "channel_density": 8192, "die_density": None, "stack_height": None,
    "dq": 64, "channel_width": 128, "pseudochannel": 2, "sid": 1, "bankgroup": 4, "bank": 4,
    "row": 1 << 14, "column": (1 << 5) << 2,
}
ramulator.dram.HBM2.timing_presets["HBM2_1880Mbps"] = {
    "rate": 1880, "nBL": 2, "nCL": 13, "nRCDRD": 13, "nRCDWR": 11,
    "nRP": 13, "nRAS": 31, "nWR": 15, "nRTPL": 5, "nCWL": 5,
    "nWTRS": 6, "nWTRL": 8,
    "tCK_ps": 1064,
}

# TPUv6e: 1 KB rows; the densities give the refresh times TOGSim has used, tRFC 350 ns and tRFCpb 200 ns.
ramulator.dram.HBM3.org_presets["HBM3_TPUv6e"] = {
    "die_density": 16384, "channel_density": 8192, "stack_height": 8,
    "dq": 32, "channel_width": 64, "pseudochannel": 2, "sid": 1, "bankgroup": 4, "bank": 4,
    "row": 1 << 15, "column": (1 << 5) << 3,
}
ramulator.dram.HBM3.timing_presets["HBM3_TPUv6e"] = {
    "rate": 6400, "nBL": 2, "nCL": 26, "nRCDRD": 26, "nRCDWR": 14,
    "nRP": 26, "nRAS": 48, "nWR": 24, "nRTP": 6, "nCWL": 8,
    "nCCDS": 2, "nWTRS": 8, "nWTRL": 12,
    "nRRDS": 8, "nRRDL": 8, "nFAW": 24, "nPPD": 2,
    "tCK_ps": 625,
}


def make_config(dram_cls, org_preset, timing_preset, controller, addr_mapper, **overrides):
    """One HBM channel per Ramulator2 instance (TOGSim makes one per channel): FRFCFS, per-bank
    refresh, open rows, External frontend. PerBank refreshes the channel's banks one at a time, so
    tREFIpb is tREFI (3.9 us) over all the channel's banks, not one pseudo channel's."""
    org, timing = dram_cls.org_presets[org_preset], dram_cls.timing_presets[timing_preset]
    banks = org["pseudochannel"] * org["sid"] * org["bankgroup"] * org["bank"]
    dram = dram_cls(org_preset=org_preset, timing_preset=timing_preset,
                    nREFIpb=3_900_000 // (banks * timing["tCK_ps"]), **overrides)
    ctrl = controller(
        dram=dram,
        scheduler=ramulator.scheduler.FRFCFS(),
        refresh_manager=ramulator.refresh_manager.PerBank(),
        row_policy=ramulator.row_policy.Open(),
        addr_mapper=addr_mapper,
        read_buffer_size=64,
        write_buffer_size=64,
    )
    ms = ramulator.memory_system.GenericDRAM(
        clock_ratio=1,
        controllers=[ctrl],
        channel_mapper=ramulator.channel_mapper.PassThroughChannelMapper(),
    )
    return {
        "frontend": {"impl": "External", "clock_ratio": 1},
        "memory_system": ms.to_config(),
    }


def gen_hbm2_tpuv3():
    """JESD235D lists no tDQSQ at 1880 Mbps, so Ramulator cannot resolve tRTW: it is tCL + tBL + 2 - tCWL."""
    t = ramulator.dram.HBM2.timing_presets["HBM2_1880Mbps"]
    return make_config(ramulator.dram.HBM2, "HBM2_TPU", "HBM2_1880Mbps", ramulator.controller.HBM12,
                       ramulator.addr_mapper.RoBaRaCoCh(), nRTW=t["nCL"] + t["nBL"] + 2 - t["nCWL"])

def gen_hbm2_tpuv4():
    return make_config(ramulator.dram.HBM2, "HBM2_TPU", "HBM2_2400Mbps", ramulator.controller.HBM12,
                       ramulator.addr_mapper.RoBaRaCoCh())

def gen_hbm3_tpuv6e():
    return make_config(ramulator.dram.HBM3, "HBM3_TPUv6e", "HBM3_TPUv6e", ramulator.controller.HBM34,
                       ramulator.addr_mapper.MOP4CLXOR())


CONFIGS = {
    "HBM2_TPUv3.yaml":  gen_hbm2_tpuv3,
    "HBM2_TPUv4.yaml":  gen_hbm2_tpuv4,
    "HBM3_TPUv6e.yaml": gen_hbm3_tpuv6e,
}

class CompactJSONEncoder(json.JSONEncoder):
    def encode(self, o, level=0):
        indent = '  ' * level
        if isinstance(o, list):
            return '[' + ', '.join(self.encode(i, level) for i in o) + ']'
        if isinstance(o, dict):
            items = ',\n'.join(
                f'{indent}  {json.dumps(k)}: {self.encode(v, level + 1)}'
                for k, v in o.items()
            )
            return '{\n' + items + '\n' + indent + '}'
        return super().encode(o)

if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    for filename, gen_fn in CONFIGS.items():
        cfg = gen_fn()
        out_path = os.path.join(out_dir, filename)
        with open(out_path, "w") as f:
            f.write(CompactJSONEncoder().encode(cfg))
        print(f"Generated {out_path}")
