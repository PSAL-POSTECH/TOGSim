#include "Core.h"
#include "CoreTraceLog.h"
#include "Result.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>

Core::Core(uint32_t id, SimulationConfig config)
    : _id(id),
      _config(config),
      _core_cycle(0),
      _stat_dma_cycle(0),
      _num_systolic_array_per_core(config.num_systolic_array_per_core),
      _typed(id, config.num_systolic_array_per_core, config.sa_weight_buffer_depth),
      _dma(id, config.dram_req_size, config.l2d_type != L2CacheType::NOCACHE) {
  _stat_inst_count.resize(static_cast<size_t>(Opcode::COUNT), 0);
  _stat_tot_skipped_inst.resize(static_cast<size_t>(Opcode::COUNT), 0);
  _sram_capacity = (size_t)config.core_spad_size_kb * 1024;  // 0 = throttle disabled
}

// The cycles a compute runs and the arrays it is spread over are its tile's window in the unit table
// (UnitTable::window); without a unit table, its table cycles.
cycle_type Core::split_compute_cycles(const std::shared_ptr<Instruction>& inst) const {
  const cycle_type cc = inst->get_compute_cycle();
  if (!_unit_table) return cc;
  const UnitTable::Window w = _unit_table->window(inst->get_tile_id(), cc);
  inst->array_split = w.n_eff;
  return (cycle_type)w.cycles;
}

void Core::apply_due(const DueAction& a) {
  switch (a.kind) {
    case DueAction::FreeWeightSlot:
      if (_typed.release(a.token)) _issue_dirty = true;   // weight slot freed -> re-arm
      break;
    case DueAction::ComputeFree:
      _issue_dirty = true;
      break;
    case DueAction::WakeBar: {
      auto bar = a.bar;            // async load data arrived -> fire its MEMORY_BAR
      if (bar->get_opcode() == Opcode::MEMORY_BAR && !bar_released(bar)) break;
      finish_instruction(bar);
      break;
    }
    case DueAction::LoadReady:     // a load's ready latency ran out: it completes as its last data had just come
      _dma_ready_pending--;
      _dma_finished_queue.push_back(a.bar);
      break;
    case DueAction::StoreDrained:  // a store's ready latency ran out; only the core's end waited for it
      _dma_ready_pending--;
      break;
  }
}

// A MEMORY_BAR is released once every async load it waits has arrived, and only once:
// two of its loads arriving in the same cycle each schedule a WakeBar.
bool Core::bar_released(const std::shared_ptr<Instruction>& bar) {
  if (bar->finished) return false;
  for (auto& key : bar->get_wait_keys())
    if (_dma.get_tag_finish(bar->subgraph_id, key) == 0) return false;
  return true;
}

void Core::process_due_events() {
  while (!_due_events.empty() && _due_events.begin()->first <= _core_cycle) {
    apply_due(_due_events.begin()->second);
    _due_events.erase(_due_events.begin());
  }
}

// A reader of buffer versions is issued: a version whose last reader this is gives its bytes back.
void Core::release_sram(const std::shared_ptr<Instruction>& inst) {
  if (!_sram_capacity) return;
  for (int64_t id : inst->get_sram_release()) {
    auto readers = _sram_readers.find(id);
    if (readers == _sram_readers.end() || --readers->second > 0) continue;
    _sram_readers.erase(readers);
    auto it = _sram_allocs.find(id);
    if (it == _sram_allocs.end()) continue;
    _sram_used -= it->second;
    _sram_allocs.erase(it);
    _issue_dirty = true;   // freed spad bytes -> re-arm
  }
}

// A work-item ended: the bytes of its versions that no reader freed leave the spad, and so does its kernel's
// hold when it is the kernel's last work-item on this core.
void Core::release_sram(const std::shared_ptr<Tile>& tile) {
  const std::shared_ptr<SpadHold>& hold = tile->get_spad_hold();
  if (hold && hold->left && --hold->left == 0 && hold->taken) {
    _sram_used -= hold->bytes;
    _issue_dirty = true;
    core_trace_log::trace_spad_hold(_core_cycle, _id, TraceLogTag::pad15(TraceLogTag::kSpadHoldEnd), hold->bytes);
  }
  if (!_sram_capacity) return;
  for (int64_t id = tile->spad_version_first(); id < tile->spad_version_end(); id++) {
    auto it = _sram_allocs.find(id);
    if (it == _sram_allocs.end()) continue;
    _sram_used -= it->second;
    _sram_allocs.erase(it);
    _issue_dirty = true;
  }
}

bool Core::try_occupy_sram(const std::shared_ptr<Instruction>& inst) {
  if (!_sram_capacity || inst->get_sram_alloc() < 0) return true;   // untracked
  size_t F = inst->sram_footprint();
  if (_sram_used + F > _sram_capacity) return false;                // would overflow -> stall
  _sram_used += F;
  _sram_allocs[inst->get_sram_alloc()] += F;                        // accumulate version footprint
  return true;
}

// Two work-items in flight when two of `op` fit the spad beside its kernel's hold (twice the larger of its
// footprint and what it needs at once), else one.
int Core::max_dispatch(const std::shared_ptr<Tile>& op) const {
  if (!_sram_capacity) return 2;
  const size_t held = op->get_spad_hold() ? op->get_spad_hold()->bytes : 0;
  const size_t room = _sram_capacity > held ? _sram_capacity - held : 0;
  return 2 * std::max(op->get_spad_footprint(), op->get_spad_at_once()) > room ? 1 : 2;
}

// A work-item enters while fewer than max_dispatch(op) are in flight. They are all of one kernel: a partition's
// kernels run one after another (TileGraph::is_finished), so a kernel's hold has ended before the next one enters.
bool Core::can_issue(const std::shared_ptr<Tile>& op) {
  return (int)_tiles.size() < max_dispatch(op);
}

// The first work-item of a kernel to reach the core takes the bytes the kernel holds on it.
void Core::issue(std::shared_ptr<Tile> op) {
  const std::shared_ptr<SpadHold>& hold = op->get_spad_hold();
  if (_sram_capacity && hold && hold->bytes && !hold->taken) {
    hold->taken = true;
    _sram_used += hold->bytes;
    core_trace_log::trace_spad_hold(_core_cycle, _id, TraceLogTag::pad15(TraceLogTag::kSpadHold), hold->bytes);
  }
  if (op->get_instructions().size())
    core_trace_log::trace_tile_scheduled(_core_cycle, _id,
                                         TraceLogTag::pad15(TraceLogTag::kTileScheduled),
                                         op->get_spad_footprint(), max_dispatch(op));
  for (const auto& inst : op->get_instructions()) {
    inst->set_owner_dirty(&_issue_dirty);   // dep-resolved enqueues re-arm THIS core
    if (_sram_capacity)
      for (int64_t id : inst->get_sram_release()) _sram_readers[id]++;
    if (inst->is_ready())
      op->enqueue_ready(inst);
  }
  _tiles.push_back(std::move(op));
  _issue_dirty = true;   // new dispatch -> re-arm
}

std::shared_ptr<Tile> Core::pop_finished_tile() {
  std::shared_ptr<Tile> result = std::make_unique<Tile>(Tile(Tile::Status::EMPTY));
  if (_finished_tiles.size() > 0) {
    result = std::move(_finished_tiles.front());
    _finished_tiles.pop();
  }
  return result;
}

void Core::vu_cycle() {
  bool retry = true;
  while (retry) {
    if (!_vu_compute_pipeline.empty()) {
      _stat_vu_compute_cycle++;
      if(_vu_compute_pipeline.front()->finish_cycle <= _core_cycle) {
        finish_instruction(_vu_compute_pipeline.front());
        _vu_compute_pipeline.pop();
      } else {
        retry = false;
      }
    } else {
      _stat_vu_compute_idle_cycle++;
      retry = false;
    }
  }
}

void Core::compute_cycle() {
  vu_cycle();
  _typed.cycle(_core_cycle, this, [](void* core, std::shared_ptr<Instruction>& inst) {
    static_cast<Core*>(core)->finish_instruction(inst);
  });
}

// Hand `inst` to the DMA engine, which holds it `latency` cycles before its first request.
void Core::start_dma(std::shared_ptr<Instruction> inst, uint32_t latency) {
  if (latency && core_trace_log::trace_enabled())
    core_trace_log::trace_instruction_line(_core_cycle, _id, TraceLogTag::pad15(TraceLogTag::kDmaLatency),
                                           inst->get_global_inst_id(),
                                           core_trace_log::format_dma_latency_line(*inst, latency));
  _dma.issue_tile(std::move(inst), latency);
}

void Core::dma_cycle() {
  /* Check finished dma operation */
  while(_dma_finished_queue.size()) {
    std::shared_ptr<Instruction>& instruction = _dma_finished_queue.at(0);
    assert(instruction->get_waiting_request()==0);

    /* Finish DMA read instruction */
    if (instruction->is_dma_read() && !instruction->is_async_dma())
      finish_instruction(instruction);

    /* Set tag table of async dma load */
    if (instruction->is_dma_read() && instruction->is_async_dma()) {
      auto& key = instruction->get_tag_id();
      assert(!_dma.get_tag_finish(instruction->subgraph_id, key));
      spdlog::trace(
          "[{}][Core {}] TOG async DMA response (table notify): tag_addr=0x{:016x} global_inst_id={} "
          "subgraph_id={}",
          _core_cycle,
          _id,
          static_cast<uint64_t>(static_cast<uintptr_t>(instruction->get_addr_id())),
          instruction->get_global_inst_id(),
          instruction->subgraph_id);
      _dma.set_tag_finish(instruction->subgraph_id, key);
      finish_instruction(instruction, InstFinishTraceTag::DmaRespComplete);
      for (auto & wait_inst : _dma.get_tag_waiter(instruction->subgraph_id, key)) {
        _dma.mark_tag_used(instruction->subgraph_id, key);
        _due_events.emplace(_core_cycle, DueAction{DueAction::WakeBar, nullptr, wait_inst});
      }
    }
    _dma_finished_queue.erase(_dma_finished_queue.begin());
  }

  if (_dma.is_finished()) {
    /* Finish instruction when it is DMA store */
    if (_dma.get_current_inst() != nullptr) {
      std::shared_ptr<Instruction> finished_inst = std::move(_dma.get_current_inst());
      if (finished_inst->is_dma_write()) {
        /* Only DMA write operation is finished! A store is asynchronous: its work-item goes on (and its
           dispatch slot frees) once the requests are out, and its ready latency only keeps the core running,
           so it shows at the run's end, not between work-items or kernels. */
        if (_config.dma_store_ready_cycles) {
          _dma_ready_pending++;
          _due_events.emplace(_core_cycle + _config.dma_store_ready_cycles,
                              DueAction{DueAction::StoreDrained, nullptr, nullptr});
        }
        finish_instruction(finished_inst);
      } else if (finished_inst->is_dma_read() && finished_inst->is_async_dma()) {
        /* Register tag table for async dma load; see TraceLogTag::kAsyncDmaAllRequestsIssued */
        finish_instruction(finished_inst, InstFinishTraceTag::DmaIssueComplete);
      } else if(!finished_inst->is_dma_read()) {
        core_trace_log::log_error_dma_instruction_invalid(_core_cycle, _id);
        exit(EXIT_FAILURE);
      } else if (finished_inst->get_opcode() == Opcode::MEMORY_BAR) {
        if (core_trace_log::trace_enabled()) core_trace_log::trace_instruction_line(_core_cycle,
                                               _id,
                                               TraceLogTag::pad15(TraceLogTag::kInstructionFinished),
                                               finished_inst->get_global_inst_id(),
                                               core_trace_log::format_instruction_detail_line(
                                                   *finished_inst));
      }
      /*Pass to waiting queue */
      _dma_waiting_queue[finished_inst.get()] = std::move(finished_inst);
    }

    /* Issue new DMA operation */
    if (!_ld_inst_queue.empty()) {
      start_dma(_ld_inst_queue.front(), _config.dma_load_latency);
      _ld_inst_queue.pop();
    } else if (!_st_inst_queue.empty()) {
      start_dma(_st_inst_queue.front(), _config.dma_store_latency);
      _st_inst_queue.pop();
    } else {
      /* DMA is idle */
      _stat_dma_idle_cycle++;
      return;
    }
  }
  /* Generate memfetch */
  const bool holding = _dma.holding();
  auto access_vec = _dma.get_memory_access(_core_cycle, _config.icnt_injection_ports_per_core);
  for (auto access : *access_vec) {
    access->set_start_cycle(_core_cycle);
    _request_queue.push(access);
  }

  /* Increase dma stat cycle */
  if (holding)
    _stat_dma_idle_cycle++;
  else
    _stat_dma_cycle++;
}

void Core::cycle() {
  /* Run compute unit and DMA unit */
  compute_cycle();
  dma_cycle();

  /* Increase core cycle counter */
  _core_cycle++;

  process_due_events();  // weight-slot frees + DMA-arrival wakeups due this cycle

  /* Iterate tile while an instruction is issued */
  bool issued = false;

  // Re-arm gate: skip the scan unless _issue_dirty was set since the last scan (a
  // ready-set grow or a resource free; else it re-walks the same blocked instructions
  // and issues nothing). Issue-identical.
  if (_issue_dirty) {
  _issue_dirty = false;

  for (int i=0; i<_tiles.size() && !issued; i++) {
    bool held_compute = false;
    auto& instructions = _tiles[i]->get_ready_instructions();
    // Resume after the prefix already known to be blocked (Tile::scan_from).
    for (auto it=_tiles[i]->scan_from(_sram_used, _typed.weight_free()); it!=instructions.end();) {
      auto& inst = *it;

      switch (inst->get_opcode()) {
        case Opcode::MOVIN:
          if (!try_occupy_sram(inst)) break;
          if (core_trace_log::trace_enabled()) core_trace_log::trace_instruction_line(_core_cycle,
                                                   _id,
                                                   TraceLogTag::pad15(TraceLogTag::kInstructionIssued),
                                                   inst->get_global_inst_id(),
                                                   core_trace_log::format_dma_inst_issued_trace_line(
                                                       *inst));
          _dma.register_tag(inst->subgraph_id, inst->get_tag_id());
          _ld_inst_queue.push(inst);
          issued = true;
          break;
        case Opcode::MOVOUT:
          release_sram(inst);   // store issued -> free the tiles it drained
          if (core_trace_log::trace_enabled()) core_trace_log::trace_instruction_line(_core_cycle,
                                                   _id,
                                                   TraceLogTag::pad15(TraceLogTag::kInstructionIssued),
                                                   inst->get_global_inst_id(),
                                                   core_trace_log::format_dma_inst_issued_trace_line(
                                                       *inst));
          _st_inst_queue.push(inst);
          issued = true;
          break;
        case Opcode::COMP:
          {
            const int ct = inst->get_compute_type();
            if (inst->get_compute_cycle() > 0 && _core_cycle < _compute_busy_until) {
              held_compute = true;
              if (!inst->held_for_core) _stat_compute_stalls++;
              inst->held_for_core = true;
              break;
            }
            // a fresh-output compute occupies its spad bytes on issue; stall if full.
            if (!try_occupy_sram(inst)) break;
            if (typed::is_typed(ct) && !_typed.admit(inst)) break;   // no free weight slot -> stall (retry)
            inst->set_compute_cycle(split_compute_cycles(inst));
            auto& target_pipeline = typed::is_typed(ct) ? _typed.pipeline(inst) : _vu_compute_pipeline;
            if (inst->get_compute_cycle() > 0 && !target_pipeline.empty()
                && target_pipeline.back()->finish_cycle > _core_cycle) {
              spdlog::error("[TOGSim] Core {} issued a compute at {} into a pipeline busy until {}",
                            _id, _core_cycle, target_pipeline.back()->finish_cycle);
              exit(EXIT_FAILURE);
            }
            inst->finish_cycle = _core_cycle + inst->get_compute_cycle();
            record_unit_window(inst);
            // release the occupancy (ISSUE) dependents so a successor overlaps this op.
            inst->fire(DepEvent::ISSUE);

            if (typed::is_typed(ct))
              if (auto token = _typed.token_to_free(inst))
                _due_events.emplace(inst->finish_cycle, DueAction{DueAction::FreeWeightSlot, token, nullptr});

            release_sram(inst);   // free the tiles it read (before the skip path)
            if (inst->get_compute_cycle() == 0) {
              inst->finish_instruction();
              count_unit_admitted(inst);
              static_cast<Tile*>(inst->get_owner())->inc_finished_inst();
              _stat_tot_skipped_inst.at(static_cast<size_t>(inst->get_opcode()))++;
              if (_tiles[i]->is_scan_cursor(it)) _tiles[i]->drop_scan_cursor();
              it = instructions.erase(it);   // erase returns the next iterator; the
              continue;                      // old code fell through to it++ on the
                                             // erased (invalidated) iterator -> UB
            } else {
              if (core_trace_log::trace_enabled()) core_trace_log::trace_instruction_line(_core_cycle,
                                                       _id,
                                                       TraceLogTag::pad15(
                                                           TraceLogTag::kInstructionIssued),
                                                       inst->get_global_inst_id(),
                                                       core_trace_log::format_instruction_detail_line(
                                                           *inst));
              target_pipeline.push(inst);
              _compute_busy_until = inst->finish_cycle;
              _due_events.emplace(inst->finish_cycle, DueAction{DueAction::ComputeFree, nullptr, nullptr});
              issued = true;
              if (typed::is_typed(ct)) _typed.count_issued(ct);
            }
          }
          break;
        case Opcode::MEMORY_BAR:
          {
            auto& keys = inst->get_wait_keys();
            bool sparse = false, pending = false;
            for (auto& key : keys) {
              uint32_t finished = _dma.get_tag_finish(inst->subgraph_id, key);
              sparse |= finished == (uint32_t)-1;
              pending |= finished == 0;
            }
            if (!pending && sparse) typed::Units::skip_matmuls_behind(inst);
            for (auto& key : keys) {
              uint32_t finished = _dma.get_tag_finish(inst->subgraph_id, key);
              if (finished == 0) _dma.register_tag_waiter(inst->subgraph_id, key, inst);
              else if (finished != (uint32_t)-1) _dma.mark_tag_used(inst->subgraph_id, key);
            }
            if (!pending) finish_instruction(inst);
            if (core_trace_log::trace_enabled()) core_trace_log::trace_instruction_line(_core_cycle,
                                                     _id,
                                                     TraceLogTag::pad15(
                                                         TraceLogTag::kInstructionIssued),
                                                     inst->get_global_inst_id(),
                                                     core_trace_log::format_instruction_detail_line(
                                                         *inst));
            issued = true;
          }
          break;
        default:
          core_trace_log::log_error_undefined_opcode();
          exit(EXIT_FAILURE);
      }

      if (issued) {
        _stat_inst_count.at(static_cast<size_t>(inst->get_opcode()))++;
        if (_tiles[i]->is_scan_cursor(it)) _tiles[i]->drop_scan_cursor();
        instructions.erase(it);
        break;
      }
      // Did not issue -> blocked on spad or a weight slot (the only two stalls).
      _tiles[i]->note_blocked(it, _sram_used, _typed.weight_free());
      it++;
    }
    if (held_compute) _tiles[i]->drop_scan_cursor();
  }

  // Keep dirty after an issue: the scan breaks at the first issue (!issued loop
  // guard), so ready instructions in later tiles were not scanned this cycle. Needed
  // for that early-break even when the issue woke no new dependent.
  if (issued) _issue_dirty = true;
  }  // if (_issue_dirty)

  /* Remove finshed tiles */
  bool retry = true;
  while (retry) {
    for (int i=0; i<_tiles.size() && !issued; i++) {
      if (_tiles[i]->all_insts_finshed()) {
        _tiles[i]->set_status(Tile::Status::FINISH);
        release_sram(_tiles[i]);
        _finished_tiles.push(std::move(_tiles[i]));
        _tiles.erase(_tiles.begin() + i); // FIXME. Inefficient data structure
        /* Let's retry */
        break;
      }
    }
    retry = false;
  }
  if(_config.core_print_interval && _core_cycle % _config.core_print_interval == 0) {
    print_current_stats();
  }
}

void Core::finish_instruction(std::shared_ptr<Instruction>& inst, InstFinishTraceTag tag) {
  if (tag == InstFinishTraceTag::DmaRespComplete) {
    if (!inst->finished) {
      core_trace_log::log_error_dram_responses_trace_not_finished(_core_cycle, _id);
      exit(EXIT_FAILURE);
    }
    if (core_trace_log::trace_enabled()) core_trace_log::trace_instruction_line(_core_cycle,
                                             _id,
                                             TraceLogTag::pad15(TraceLogTag::kAllDramResponsesReceived),
                                             inst->get_global_inst_id(),
                                             core_trace_log::format_instruction_detail_line(*inst));
    return;
  }
  if (inst->finished) {
    core_trace_log::log_error_instruction_already_finished(_core_cycle, _id,
                                                           opcode_to_string(inst->get_opcode()));
    exit(EXIT_FAILURE);
  }
  inst->finish_instruction();
  count_unit_admitted(inst);
  static_cast<Tile*>(inst->get_owner())->inc_finished_inst();
  const char* trace_tag = (tag == InstFinishTraceTag::DmaIssueComplete)
                              ? TraceLogTag::kAsyncDmaAllRequestsIssued
                              : TraceLogTag::kInstructionFinished;
  if (core_trace_log::trace_enabled()) core_trace_log::trace_instruction_line(_core_cycle,
                                           _id,
                                           TraceLogTag::pad15(trace_tag),
                                           inst->get_global_inst_id(),
                                           core_trace_log::format_instruction_detail_line(*inst));
}

bool Core::has_inflight() {
  // running() without the "_tiles.size() > 0" term: work that will produce a
  // finish event on its own (so the sim is NOT frozen). If this is false but
  // tiles remain, only stalled ready instructions are left.
  if (!_vu_compute_pipeline.empty()) return true;
  if (_typed.busy()) return true;
  if (!_dma_waiting_queue.empty() || !_dma_finished_queue.empty()) return true;
  if (_dma_ready_pending) return true;
  if (!_dma.empty()) return true;
  if (!_ld_inst_queue.empty() || !_st_inst_queue.empty()) return true;
  return false;
}

bool Core::running() {
  bool running = false;
  running = running || _tiles.size() > 0;
  running = running || !_vu_compute_pipeline.empty();
  running = running || _typed.busy();
  running = running || !_dma_waiting_queue.empty() || !_dma_finished_queue.empty();
  running = running || _dma_ready_pending;
  running = running || !_dma.empty();
  running = running || !_ld_inst_queue.empty();
  running = running || !_st_inst_queue.empty();
  return running;
}

bool Core::has_memory_request() {
  return !_request_queue.empty();
}

void Core::pop_memory_request() {
  _request_queue.pop();
}

void Core::push_memory_response(mem_fetch* response) {
  Instruction* owner_inst = static_cast<Instruction*>(response->get_custom_data());
  assert(owner_inst->get_waiting_request());

  if (!owner_inst->got_first_response()) {   // first data of this load arrived
    owner_inst->mark_first_response();
    if (core_trace_log::trace_enabled()) core_trace_log::trace_instruction_line(_core_cycle, _id,
        TraceLogTag::pad15(TraceLogTag::kFirstDramResponse),
        owner_inst->get_global_inst_id(),
        core_trace_log::format_instruction_detail_line(*owner_inst));
  }
  owner_inst->dec_waiting_request();
  if (!owner_inst->get_waiting_request()) {
    auto it = _dma_waiting_queue.find(owner_inst);
    if (it != _dma_waiting_queue.end()) {
      std::shared_ptr<Instruction> moved_inst = std::move(it->second);
      _dma_waiting_queue.erase(it);
      if (moved_inst->is_dma_read() && _config.dma_load_ready_cycles) {
        // A load completes its ready latency after its last data: waited, not occupied, so the latencies of
        // loads in flight together overlap and a chain of loads pays one each (tpu-perf-archive data/v6e/dma).
        _dma_ready_pending++;
        _due_events.emplace(_core_cycle + _config.dma_load_ready_cycles,
                            DueAction{DueAction::LoadReady, nullptr, std::move(moved_inst)});
      } else
        _dma_finished_queue.push_back(std::move(moved_inst));
    } else {
      assert(true || "Can't happend...!");
    }
  }
  _stat_mem_response++;
  delete response;
}

bool Core::can_issue_compute(std::shared_ptr<Instruction>& inst) {
  return inst->is_ready();
}

void Core::print_stats() {
  update_stats();
  spdlog::info("===== Instructions count =====");
  for (int i = 0; i < static_cast<size_t>(Opcode::COUNT); i++) {
    auto opcode  = static_cast<Opcode>(i);
    auto inst = _stat_inst_count.at(i);
    auto skipped = _stat_tot_skipped_inst.at(i);
    auto name = opcode_to_string(opcode);

    if (opcode == Opcode::COMP) {
      auto gemm   = _typed.gemm_inst();
      auto xlu    = _typed.xlu_inst();
      auto vector = inst - gemm - xlu;
      if (skipped)
        spdlog::info("Core [{}] : {:8} inst_count: {} (GEMM: {}, Vector: {}, XLU: {}), skipped inst_count {}",
            _id, name, inst, gemm, vector, xlu, skipped);
      else
        spdlog::info("Core [{}] : {:8} inst_count: {} (GEMM: {}, Vector: {}, XLU: {})",
            _id, name, inst, gemm, vector, xlu);
    }
    else {
      if (skipped)
        spdlog::info("Core [{}] : {:8} inst_count: {}, skipped inst_count: {}",
            _id, name, inst, skipped);
      else
        spdlog::info("Core [{}] : {:8} inst_count: {}",
            _id, name, inst);
    }
  }
  spdlog::info("========= Core stat =========");
  bool units_ok = true;
  if (_unit_table) {
    for (size_t u = 0; u < _unit_table->num_units(); u++)
      units_ok = ((int64_t)u == _unit_table->array_unit()
                    ? _unit_table->print_spread(fmt::format("Core [{}]", _id), u, _stat_tot_unit_credit.at(u), _core_cycle)
                    : _unit_table->print(fmt::format("Core [{}]", _id), u, _stat_tot_unit_admitted.at(u), _core_cycle, true))
                 && units_ok;
  } else {
    _typed.print_array_stats(_core_cycle, true);
  }
  float dram_bw = _config.dram_req_size * _stat_tot_mem_response * _config.core_freq_mhz / (_core_cycle * 1000); // B/cycle
  spdlog::info("Core [{}] : DMA active_cycles: {}, DMA idle_cycles: {}, DRAM BW: {:.3f} GB/s ({} responses)", _id, _stat_tot_dma_cycle, _stat_tot_dma_idle_cycle, dram_bw, _stat_tot_mem_response);
  result().dma(_id, {_stat_tot_dma_cycle, _stat_tot_dma_idle_cycle, dram_bw, _stat_tot_mem_response});
  if (!_unit_table) {
    spdlog::info("Core [{}] : Vector unit utilization(%): {:.2f}, active cycle: {}, idle_cycle: {}", _id,
      static_cast<float>(_stat_tot_vu_compute_cycle * 100) / _core_cycle, _stat_tot_vu_compute_cycle, _stat_tot_vu_compute_idle_cycle);
    result().unit(fmt::format("Core [{}]", _id), "Vector unit",
                  static_cast<float>(_stat_tot_vu_compute_cycle * 100) / _core_cycle, _stat_tot_vu_compute_cycle,
                  _stat_tot_vu_compute_idle_cycle);
    _typed.print_cross_lane_stats(_core_cycle, true);
  }
  spdlog::info("Core [{}] : NUMA local memory: {} requests, remote memory: {} requests", _id, _stat_numa_local_access, _stat_numa_remote_access);
  spdlog::info("Core [{}] : Total_cycles: {}", _id, _core_cycle);
  result().core_cycles(_id, _core_cycle);
  spdlog::info("Core [{}] : computes held at least once by the core's compute in flight: {}", _id, _stat_compute_stalls);
  if (!units_ok) {
    spdlog::error("[TOGSim] --unit_table: a unit admitted more than its capacity allows");
    exit(EXIT_FAILURE);
  }
  if (_unit_table && _config.core_print_interval) {
    credit_unit_windows(_unit_spread_from, _core_cycle);
    bool spread_ok = _unit_windows.empty();
    if (!spread_ok)
      spdlog::error("Core [{}] : {} compute windows end after the run's {} cycles", _id, _unit_windows.size(), _core_cycle);
    for (size_t u = 0; u < _unit_table->num_units(); u++) {
      const double spread = _stat_tot_unit_spread.at(u) + _stat_unit_spread.at(u);
      const double admitted = _stat_tot_unit_credit.at(u);
      if (std::fabs(spread - admitted) <= 1e-6 * std::max(1.0, admitted)) continue;
      spdlog::error("Core [{}] : {} periodic credit sums to {} but the run admitted {}", _id, _unit_table->unit_name(u), spread, admitted);
      spread_ok = false;
    }
    if (!spread_ok) {
      spdlog::error("[TOGSim] --unit_table: periodic unit credit does not add up to the run's admissions");
      exit(EXIT_FAILURE);
    }
  }
}

void Core::record_unit_window(const std::shared_ptr<Instruction>& inst) {
  if (!_unit_table || !_config.core_print_interval || !_unit_table->has_rows(inst->get_tile_id())) return;
  const cycle_type cc = inst->get_compute_cycle();
  if (cc == 0) {
    _unit_table->accumulate(inst->get_tile_id(), 1.0, _stat_unit_spread, inst->array_split);
    return;
  }
  _unit_windows.push_back({inst->finish_cycle - cc, inst->finish_cycle, inst->get_tile_id(), inst->array_split});
}

void Core::credit_unit_windows(cycle_type lo, cycle_type hi) {
  for (const auto& w : _unit_windows) {
    const cycle_type a = std::max(w.start, lo), b = std::min(w.end, hi);
    if (b > a) _unit_table->accumulate(w.tile_id, (double)(b - a) / (double)(w.end - w.start), _stat_unit_spread, w.n_eff);
  }
  _unit_windows.erase(std::remove_if(_unit_windows.begin(), _unit_windows.end(),
                                     [hi](const UnitWindow& w) { return w.end <= hi; }),
                      _unit_windows.end());
}

void Core::print_current_stats() {
  float dram_bw = _config.dram_req_size * _stat_mem_response * _config.core_freq_mhz / (_config.core_print_interval * 1000); // B/cycle
  auto level = spdlog::level::info;
  if(_id != 0)
    level = spdlog::level::debug;

  spdlog::info("========= Core stat =========");
  bool spread_ok = true;
  if (_unit_table) {
    credit_unit_windows(_unit_spread_from, _core_cycle);
    for (size_t u = 0; u < _unit_table->num_units(); u++) {
      spread_ok = _unit_table->print_spread(fmt::format("Core [{}]", _id), u, _stat_unit_spread.at(u), _config.core_print_interval) && spread_ok;
      _stat_tot_unit_spread.at(u) += _stat_unit_spread.at(u);
      _stat_unit_spread.at(u) = 0.0;
    }
    _unit_spread_from = _core_cycle;
  } else {
    _typed.print_array_stats(_config.core_print_interval, false);
  }
  spdlog::info("Core [{}] : DMA active_cycles: {}, DMA idle_cycles: {}, DRAM BW: {:.3f} GB/s ({} responses)", _id, _stat_dma_cycle, _stat_dma_idle_cycle, dram_bw, _stat_mem_response);
  if (!_unit_table) {
    spdlog::info("Core [{}] : Vector unit Utilization(%): {:.2f}, active_cycles: {}, idle_cycles: {}", _id,
      static_cast<float>(_stat_vu_compute_cycle * 100) / _config.core_print_interval, _stat_vu_compute_cycle, _stat_vu_compute_idle_cycle);
    _typed.print_cross_lane_stats(_config.core_print_interval, false);
  }
  spdlog::info("Core [{}] : Total_cycles: {}", _id, _core_cycle);
  update_stats();
  if (!spread_ok) {
    spdlog::error("[TOGSim] --unit_table: a unit's periodic credit exceeds its capacity");
    exit(EXIT_FAILURE);
  }
}

void Core::update_stats() {
  _typed.roll_stats();
  _stat_tot_vu_compute_cycle += _stat_vu_compute_cycle;
  _stat_tot_dma_cycle += _stat_dma_cycle;
  _stat_tot_dma_idle_cycle += _stat_dma_idle_cycle;
  _stat_tot_mem_response += +_stat_mem_response;
  for (size_t u = 0; u < _stat_unit_admitted.size(); u++) {
    _stat_tot_unit_admitted[u] += _stat_unit_admitted[u];
    _stat_unit_admitted[u] = 0;
  }

  _stat_vu_compute_cycle = 0;
  _stat_dma_cycle = 0;
  _stat_dma_idle_cycle = 0;
  _stat_vu_compute_idle_cycle = 0;
  _stat_mem_response = 0;
}