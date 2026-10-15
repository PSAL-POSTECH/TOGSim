#pragma once
#include <unordered_set>
#include <map>
#include <memory>
#include <vector>
#include <fmt/core.h>
#include <fmt/ranges.h>

#include "Dram.h"
#include "Tile.h"
#include "SimulationConfig.h"
#include "DMA.h"
#include "TraceLogTags.h"
#include "UnitTable.h"

/** Log tag kind for Core::finish_instruction (see TraceLogTag names in TraceLogTags.h). */
enum class InstFinishTraceTag {
  Fnshed,
  DmaIssueComplete,
  DmaRespComplete,
};

// A timed effect due at a cycle: free a weight slot, wake a MEMORY_BAR, or free the core's compute.
struct DueAction {
  enum Kind { FreeWeightSlot, WakeBar, ComputeFree } kind;
  std::shared_ptr<typed::WeightToken> token;
  std::shared_ptr<Instruction> bar;
};

class Core {
 public:
  Core(uint32_t id, SimulationConfig config);
  ~Core()=default;
  virtual bool running();
  // True if this core has work actively in flight (DMA / compute pipeline / queues)
  // that will produce a future finish event -- i.e. running() minus "tiles waiting".
  // Used by the frozen-state (spad-too-small) guard.
  bool has_inflight();
  virtual bool can_issue(const std::shared_ptr<Tile>& op);
  int max_dispatch(const std::shared_ptr<Tile>& op) const;
  virtual void issue(std::shared_ptr<Tile> tile);
  virtual std::shared_ptr<Tile> pop_finished_tile();
  virtual void cycle();
  virtual void print_stats();
  virtual void print_current_stats();
  virtual void finish_instruction(std::shared_ptr<Instruction>& inst,
                                  InstFinishTraceTag tag = InstFinishTraceTag::Fnshed);
  virtual bool has_memory_request();
  virtual void pop_memory_request();
  virtual mem_fetch* top_memory_request() { return _request_queue.front(); }
  virtual void push_memory_response(mem_fetch* response);
  void check_tag() { _dma.check_table(); }
  bool bar_released(const std::shared_ptr<Instruction>& bar);
  void inc_numa_local_access() { _stat_numa_local_access++; }
  void inc_numa_remote_access() { _stat_numa_remote_access++; }
  // --unit_table: sum each completed compute's per-unit admissions on this core.
  void set_unit_table(const UnitTable* t) {
    _unit_table = t;
    _stat_unit_admitted.resize(t->num_units(), 0);
    _stat_tot_unit_admitted.resize(t->num_units(), 0);
    _stat_unit_spread.resize(t->num_units(), 0.0);
    _stat_tot_unit_spread.resize(t->num_units(), 0.0);
    _stat_tot_unit_credit.resize(t->num_units(), 0.0);
  }
  const std::vector<uint64_t>& get_tot_unit_admitted() const { return _stat_tot_unit_admitted; }
  const std::vector<double>& get_tot_unit_credit() const { return _stat_tot_unit_credit; }
  cycle_type get_core_cycle() const { return _core_cycle; }


 protected:
  void dma_cycle();
  void compute_cycle();
  void vu_cycle();
  bool can_issue_compute(std::shared_ptr<Instruction>& inst);
  void update_stats();
  // Called once per finished instruction; counts a compute's tile into the unit table, and its
  // per-array credit (the array unit over the compute's N_eff) into _stat_tot_unit_credit.
  void count_unit_admitted(const std::shared_ptr<Instruction>& inst) {
    if (!_unit_table || inst->get_opcode() != Opcode::COMP) return;
    _unit_table->accumulate(inst->get_tile_id(), _stat_unit_admitted);
    _unit_table->accumulate(inst->get_tile_id(), 1.0, _stat_tot_unit_credit, inst->array_split);
  }
  // Periodic --unit_table credit: a compute with unit rows spreads its tile's admissions
  // evenly over its execution window [finish - compute_cycle, finish); a zero-length
  // window credits the interval being accumulated now.
  void record_unit_window(const std::shared_ptr<Instruction>& inst);
  // Credit every recorded window's share of [lo, hi) into _stat_unit_spread; drop windows ended by hi.
  void credit_unit_windows(cycle_type lo, cycle_type hi);
  // A consumer leaves the buffer versions it read; the last reader of a version to be issued frees its bytes.
  void release_sram(const std::shared_ptr<Instruction>& inst);
  void release_sram(const std::shared_ptr<Tile>& tile);
  // Occupy inst's buffer-version footprint on issue; false if it would overflow
  // the spad this cycle (the caller stalls it). True for untracked insts.
  bool try_occupy_sram(const std::shared_ptr<Instruction>& inst);
  // The cycles a compute runs: its table cycles, shortened under systolic_array_split by
  // spreading its array unit's rows over the arrays (see SimulationConfig::systolic_array_split).
  cycle_type split_compute_cycles(const std::shared_ptr<Instruction>& inst) const;
  void process_due_events();   // drain _due_events due this cycle
  void start_dma(std::shared_ptr<Instruction> inst, uint32_t latency);
  void apply_due(const DueAction& a);

  /* Core id & config file */
  const uint32_t _id;
  const SimulationConfig _config;
  uint32_t _num_systolic_array_per_core;
  // DEPRECATED typed grouping: the systolic-array and cross-lane pipelines and the weight slots.
  typed::Units _typed;

  /* DMA Unit */
  DMA _dma;

  /* cycle */
  cycle_type _core_cycle;
  cycle_type _stat_tot_vu_compute_cycle = 0;
  cycle_type _stat_tot_dma_cycle = 0;
  cycle_type _stat_tot_dma_idle_cycle = 0;
  cycle_type _stat_tot_vu_compute_idle_cycle = 0;
  std::vector<uint64_t> _stat_inst_count;
  std::vector<uint64_t> _stat_tot_skipped_inst;
  uint64_t _stat_tot_mem_response = 0;
  uint64_t _stat_compute_stalls = 0;   // computes held at least once by the compute in flight
  uint64_t _stat_skip_dma = 0;
  uint64_t _stat_numa_local_access = 0;
  uint64_t _stat_numa_remote_access = 0;
  const UnitTable* _unit_table = nullptr;
  std::vector<uint64_t> _stat_unit_admitted;
  std::vector<uint64_t> _stat_tot_unit_admitted;
  struct UnitWindow { cycle_type start, end; int64_t tile_id; uint32_t n_eff; };
  std::vector<double> _stat_tot_unit_credit;   // per-array admissions of the finished computes
  std::vector<UnitWindow> _unit_windows;
  std::vector<double> _stat_unit_spread;       // this interval's time-spread credit
  std::vector<double> _stat_tot_unit_spread;   // summed over the printed intervals
  cycle_type _unit_spread_from = 0;            // start of the interval being accumulated

  cycle_type _stat_vu_compute_cycle = 0;
  cycle_type _stat_dma_cycle = 0;
  cycle_type _stat_dma_idle_cycle = 0;
  cycle_type _stat_vu_compute_idle_cycle = 0;
  uint64_t _stat_mem_response = 0;

  std::vector<std::shared_ptr<Tile>> _tiles;
  std::queue<std::shared_ptr<Tile>> _finished_tiles;

  // Issue-scan re-arm (perf): cycle() skips the ready-queue scan unless this is set.
  // EVERY event that can make a stalled instruction issuable must set it -- a new
  // issue-gating throttle that forgets to will make cycle() skip the scan forever.
  bool _issue_dirty = true;

  // One compute at a time per core: a compute with cycles is held until the previous one finished
  // (gem5 issues in order; a pop waits for its row). The tile drops its scan cursor; ComputeFree
  // at finish re-arms the scan. cc==0 computes are exempt; DMA is not serialized, by design.
  cycle_type _compute_busy_until = 0;

  std::queue<std::shared_ptr<Instruction>> _vu_compute_pipeline;
  std::queue<std::shared_ptr<Instruction>> _ld_inst_queue;
  std::queue<std::shared_ptr<Instruction>> _st_inst_queue;

  std::unordered_map<Instruction*, std::shared_ptr<Instruction>> _dma_waiting_queue;
  std::vector<std::shared_ptr<Instruction>> _dma_finished_queue;
  /* Interconnect queue */
  std::queue<mem_fetch*> _request_queue;
  std::queue<mem_fetch*> _response_queue;
  uint32_t _waiting_write_reqs;

  // The spad: bytes in use, the limit (0: none), and each buffer version's bytes.
  size_t _sram_used = 0;
  size_t _sram_capacity = 0;
  std::unordered_map<int64_t, size_t> _sram_allocs;
  std::unordered_map<int64_t, int> _sram_readers;
  std::vector<std::shared_ptr<SpadHold>> _spad_holds;
  std::multimap<cycle_type, DueAction> _due_events;
};