#pragma once
// Result -- a run's results as data (--result <path>): the values the end-of-run lines print, each
// recorded where it is printed, written as one JSON object when the run ends. The lines are for a
// reader; this file is for a program, which then needs no pattern over them.
#include <cstdint>
#include <map>
#include <string>
#include <vector>

class Result {
 public:
  struct Unit {
    double utilization_pct;
    int64_t active_cycles;
    int64_t idle_cycles;
  };
  struct Dma {
    uint64_t active_cycles;
    uint64_t idle_cycles;
    double dram_bw_gbps;
    uint64_t responses;
  };
  struct Channel {
    double bandwidth_gbps;
    double utilization_pct;
    long long reads;
    long long writes;
  };
  struct Kernel {
    int scheduler;
    int kernel_id;
    uint64_t started;
    uint64_t finished;
  };

  // `who` is "Core [i]" or "Total", as the line names it; a later record of the same unit replaces
  // an earlier one, so the run's totals, printed last, are what the file holds.
  void unit(const std::string& who, const std::string& name, double utilization_pct, int64_t active,
            int64_t idle);
  void dma(int core, const Dma& dma) { _dma[core] = dma; }
  void core_cycles(int core, uint64_t cycles) { _core_cycles[core] = cycles; }
  void dram_channel(int channel, const Channel& c) { _channels[channel] = c; }
  void dram_combined(const Channel& c) { _combined = c; _has_combined = true; }
  void kernel(const Kernel& k) { _kernels.push_back(k); }
  void total_cycles(uint64_t cycles) { _total_cycles = cycles; }
  void wall_clock(double seconds) { _wall_clock_s = seconds; }

  // Write the JSON object to `path`; on failure, log it and exit(1).
  void write(const std::string& path) const;

 private:
  std::map<std::string, std::map<std::string, Unit>> _units;
  std::map<int, Dma> _dma;
  std::map<int, uint64_t> _core_cycles;
  std::map<int, Channel> _channels;
  Channel _combined{};
  bool _has_combined = false;
  std::vector<Kernel> _kernels;
  uint64_t _total_cycles = 0;
  double _wall_clock_s = -1;
};

// The run's one Result, which the printers record into.
Result& result();
