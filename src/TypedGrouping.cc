// TypedGrouping.cc -- see TypedGrouping.h. DEPRECATED with the typed grouping: kept so a typed TOG
// still replays for comparison, and removed with it.
#include "TypedGrouping.h"

#include <spdlog/spdlog.h>

#include "Instruction.h"

namespace typed {

Units::Units(uint32_t core_id, uint32_t num_arrays, uint32_t weight_slot_depth)
    : _core_id(core_id), _num_arrays(num_arrays), _weight_slot_depth(weight_slot_depth) {
  _sa_pipeline.resize(_num_arrays);
  _stat_tot_sa_cycle.resize(_num_arrays);
  _stat_sa_cycle.resize(_num_arrays);
  _stat_tot_sa_idle_cycle.resize(_num_arrays);
  _stat_sa_idle_cycle.resize(_num_arrays);
  if (_weight_slot_depth == 0) {
    spdlog::error("sa_weight_buffer_depth must be > 0 (raise it to loosen the preload throttle)");
    exit(EXIT_FAILURE);
  }
  _weight_slots_used.resize(_num_arrays, 0);
  _weight_free = _num_arrays * (int)_weight_slot_depth;
}

// Round-robin a systolic array that still has a free weight slot; -1 if all full
// (the preload must stall). Advances _array_rr past the chosen array.
int Units::pick_free_weight_sa() {
  for (uint32_t i = 0; i < _num_arrays; i++) {
    uint32_t s = (_array_rr + i) % _num_arrays;
    if (_weight_slots_used[s] < (int)_weight_slot_depth) {
      _array_rr = (s + 1) % _num_arrays;
      return (int)s;
    }
  }
  return -1;
}

// A preload's weight-slot refcount: the matmuls subscribed to its ISSUE event. One preload can
// have 300k subscribers and the issue scan asks on every stalled cycle, so it is counted once.
int Units::matmul_consumers(const std::shared_ptr<Instruction>& preload) {
  int& n = preload->typed.n_matmul_consumers;
  if (n < 0) {
    n = 0;
    for (auto& c : preload->get_deps(DepEvent::ISSUE))
      if (c->get_compute_type() == MATMUL) n++;
  }
  return n;
}

// A preload takes an array with a free weight slot and pins its matmul consumers there; a matmul
// runs on the array its preload pinned. The slot is asked for first: without one nothing else
// about the preload is worth computing.
bool Units::admit(const std::shared_ptr<Instruction>& inst) {
  const int ct = inst->get_compute_type();
  if (ct != MATMUL && ct != PRELOAD) return true;
  int sa_idx = -1;
  if (ct == PRELOAD) {
    sa_idx = pick_free_weight_sa();
    if (sa_idx < 0) return false;
    const int n_consumers = matmul_consumers(inst);
    if (n_consumers == 0) {
      spdlog::error("preload has no matmul consumer (weight-slot model invariant)");
      exit(EXIT_FAILURE);
    }
    _weight_slots_used[sa_idx]++;
    _weight_free--;
    auto tok = std::make_shared<WeightToken>(WeightToken{sa_idx, n_consumers});
    for (auto& c : inst->get_deps(DepEvent::ISSUE))
      if (c->get_compute_type() == MATMUL) {
        c->typed.assigned_sa = sa_idx;
        c->typed.weight_token = tok;
      }
  } else {
    sa_idx = inst->typed.assigned_sa;
    if (sa_idx < 0) {
      spdlog::error("matmul was not pinned to an SA by a preload (weight-slot model invariant)");
      exit(EXIT_FAILURE);
    }
  }
  inst->typed.assigned_sa = sa_idx;
  return true;
}

std::queue<std::shared_ptr<Instruction>>& Units::pipeline(const std::shared_ptr<Instruction>& inst) {
  const int ct = inst->get_compute_type();
  if (ct == CROSS_LANE) return _xlu_pipeline;
  if (ct == MATMUL || ct == PRELOAD) return _sa_pipeline.at(inst->typed.assigned_sa);
  spdlog::error("Undefined compute type");
  exit(EXIT_FAILURE);
}

std::shared_ptr<WeightToken> Units::token_to_free(const std::shared_ptr<Instruction>& inst) const {
  return inst->get_compute_type() == MATMUL ? inst->typed.weight_token : nullptr;
}

bool Units::release(const std::shared_ptr<WeightToken>& token) {
  if (--token->refcount > 0) return false;
  _weight_slots_used[token->sa]--;
  _weight_free++;
  return true;
}

void Units::count_issued(int compute_type) {
  if (compute_type == MATMUL || compute_type == PRELOAD) _stat_gemm_inst++;
  else if (compute_type == CROSS_LANE) _stat_xlu_inst++;
}

void Units::skip_matmuls_behind(const std::shared_ptr<Instruction>& bar) {
  for (auto child : bar->get_deps(DepEvent::DONE))
    if (child->get_opcode() == Opcode::COMP && child->get_compute_type() == MATMUL)
      child->set_compute_cycle(0);
}

void Units::cycle(uint64_t core_cycle, void* core, Finish finish) {
  for (uint32_t i = 0; i < _num_arrays; i++) {
    bool retry = true;
    while (retry) {
      if (!_sa_pipeline.at(i).empty()) {
        if (_sa_pipeline.at(i).front()->finish_cycle <= core_cycle) {
          finish(core, _sa_pipeline.at(i).front());
          _sa_pipeline.at(i).pop();
        } else {
          _stat_sa_cycle.at(i)++;
          retry = false;
        }
      } else {
        _stat_sa_idle_cycle.at(i)++;
        retry = false;
      }
    }
  }
  bool retry = true;
  while (retry) {
    if (!_xlu_pipeline.empty()) {
      _stat_xlu_cycle++;
      if (_xlu_pipeline.front()->finish_cycle <= core_cycle) {
        finish(core, _xlu_pipeline.front());
        _xlu_pipeline.pop();
      } else {
        retry = false;
      }
    } else {
      _stat_xlu_idle_cycle++;
      retry = false;
    }
  }
}

bool Units::busy() const {
  if (!_xlu_pipeline.empty()) return true;
  for (auto& q : _sa_pipeline)
    if (!q.empty()) return true;
  return false;
}

void Units::roll_stats() {
  for (uint32_t i = 0; i < _num_arrays; i++) {
    _stat_tot_sa_cycle.at(i) += _stat_sa_cycle.at(i);
    _stat_tot_sa_idle_cycle.at(i) += _stat_sa_idle_cycle.at(i);
    _stat_sa_cycle.at(i) = 0;
    _stat_sa_idle_cycle.at(i) = 0;
  }
  _stat_tot_xlu_cycle += _stat_xlu_cycle;
  _stat_tot_xlu_idle_cycle += _stat_xlu_idle_cycle;
  _stat_xlu_cycle = 0;
  _stat_xlu_idle_cycle = 0;
}

// The run's totals over `cycles`, or the interval since the last roll_stats().
void Units::print_array_stats(uint64_t cycles, bool total) const {
  const auto& active = total ? _stat_tot_sa_cycle : _stat_sa_cycle;
  const auto& idle = total ? _stat_tot_sa_idle_cycle : _stat_sa_idle_cycle;
  for (uint32_t i = 0; i < _num_arrays; i++)
    spdlog::info("Core [{}] : Systolic array [{}] utilization(%): {:.2f}, active_cycles: {}, idle_cycles: {}", _core_id, i,
      static_cast<float>(active.at(i) * 100) / cycles, active.at(i), idle.at(i));
}

void Units::print_cross_lane_stats(uint64_t cycles, bool total) const {
  if (total)
    spdlog::info("Core [{}] : Cross-lane unit utilization(%): {:.2f}, active cycle: {}, idle_cycle: {}", _core_id,
      static_cast<float>(_stat_tot_xlu_cycle * 100) / cycles, _stat_tot_xlu_cycle, _stat_tot_xlu_idle_cycle);
  else
    spdlog::info("Core [{}] : Cross-lane unit Utilization(%): {:.2f}, active_cycles: {}, idle_cycles: {}", _core_id,
      static_cast<float>(_stat_xlu_cycle * 100) / cycles, _stat_xlu_cycle, _stat_xlu_idle_cycle);
}

}  // namespace typed
