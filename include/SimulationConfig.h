#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <vector>
#include <string>
#include <yaml-cpp/yaml.h>

enum class CoreType { WS_MESH };

enum class DramType { SIMPLE, RAMULATOR2 };

enum class IcntType { SIMPLE, BOOKSIM2 };

enum class L2CacheType { NOCACHE, DATACACHE };

struct SimulationConfig {
  /* Path to the top-level hardware YAML passed to the simulator (empty if not from a file). */
  std::string config_file_path;

  /* Core config */
  std::vector<CoreType> core_type;
  uint32_t num_cores;
  uint32_t core_freq_mhz;
  uint32_t core_print_interval = 0;
  uint32_t num_systolic_array_per_core = 1;
  // Per-core spad capacity (KB) for the DMA throttle: a load that would overflow it waits for a
  // consumer to free a tile. vpu_spad_size_kb_per_lane x vpu_num_lanes when the config gives both,
  // else the config's core_spad_size_kb (core_spad_kb in Common.cc); 0 = no limit.
  uint32_t core_spad_size_kb = 0;
  std::optional<std::vector<uint64_t>> tog_tensor_tile;
  // DEPRECATED typed grouping. SA weight-buffer depth: weight tiles a systolic array holds; a
  // preload stalls until a slot frees (its matmuls finished). 2 = weight double-buffer
  // (convention default, tunable). 0 is refused.
  uint32_t sa_weight_buffer_depth = 2;
  // systolic_array_split: round_robin (default) -- a compute's array-unit rows spread over
  // min(N, ceil(rows / vpu_num_lanes)) arrays (needs --unit_table); none ignores N.
  bool systolic_array_split = true;
  // systolic_array_unit: the name of the unit table's array unit, whatever the table calls it (SYSTOLIC, TPUV6_MXU, ...;
  // one per machine): the split reads that unit's rows, and a table without it is refused.
  // The default is the only place TOGSim names a unit: vcix's libtpu.so port calls its array SYSTOLIC.
  std::string systolic_array_unit = "SYSTOLIC";
  uint32_t vpu_num_lanes = 0;

  /* DRAM config */
  DramType dram_type;
  uint32_t dram_num_partitions = 1;
  uint32_t dram_channels_per_partitions = 0;
  uint32_t dram_freq_mhz;
  uint32_t dram_channels;
  uint32_t dram_req_size;
  uint32_t dram_latency;
  float dram_bandwidth_gbps_per_channel = 0.f;
  uint32_t dram_print_interval;
  std::string dram_config_path;

  /* L2 Cache config */
  L2CacheType l2d_type = L2CacheType::NOCACHE;
  std::string l2d_config_str;
  uint32_t l2d_hit_latency = 1;

  /* ICNT config */
  IcntType icnt_type;
  uint32_t icnt_injection_ports_per_core = 1;
  std::string icnt_config_path;
  uint32_t icnt_freq_mhz;
  uint32_t icnt_latency;
  uint32_t icnt_stats_print_period_cycles=0;

  /* Sheduler config */
  uint32_t num_partition=1;

  /* Core id, Partiton id mapping */
  std::map<uint32_t, uint32_t> partiton_map;

  uint64_t align_address(uint64_t addr) {
    return addr - (addr % dram_req_size);
  }

  float max_dram_bandwidth() const {
    if (dram_bandwidth_gbps_per_channel > 0.f)
      return dram_bandwidth_gbps_per_channel * static_cast<float>(dram_channels);
    return 0.f;
  }

  /** Resolve `path` for opening on disk: absolute paths as-is; relative paths against top-level config dir. */
  std::string resolve_against_simulation_config(const std::string& path) const {
    namespace fs = std::filesystem;
    if (path.empty())
      return path;
    fs::path p(path);
    fs::path abs = p.is_absolute() ? fs::absolute(p)
                 : !config_file_path.empty()
                     ? fs::absolute(fs::path(config_file_path).parent_path() / p)
                     : fs::absolute(p);
    std::error_code ec;
    fs::path canon = fs::weakly_canonical(abs, ec);
    return (ec ? abs : canon).string();
  }
};