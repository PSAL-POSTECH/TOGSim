#pragma once
// TypedGrouping.h -- what TOGSim models only for a TOG of the typed grouping, where the compiler
// classifies each compute (compute_type 1 matmul / 2 preload / 3 cross-lane): the per-type
// pipelines, the weight slots and the occupancy edges. A TOG whose computes are all 0 uses none.
#include <cstdint>
#include <memory>
#include <queue>
#include <vector>

class Instruction;

namespace typed {

enum ComputeType { MATMUL = 1, PRELOAD = 2, CROSS_LANE = 3 };

inline bool is_typed(int compute_type) { return compute_type != 0; }

// A matmul joins the writers of the buffer it accumulates into instead of replacing them.
inline bool accumulates(int compute_type) { return compute_type == MATMUL; }

// A matmul overlaps the matmul or preload it reads from: released at that one's issue, not its finish.
inline bool overlaps(int consumer_type, int writer_type) {
  return consumer_type == MATMUL && (writer_type == MATMUL || writer_type == PRELOAD);
}

// One weight slot on systolic array `sa`: a preload sets refcount to the matmuls reusing the
// weight, each frees it when it finishes, and the last one releases the slot.
struct WeightToken { int sa; int refcount; };

// The typed state an Instruction carries: its array, its weight slot, its matmul consumers.
struct InstState {
  int assigned_sa = -1;
  std::shared_ptr<WeightToken> weight_token;
  int n_matmul_consumers = -1;
};

// A core's systolic-array and cross-lane pipelines, its weight slots and their statistics.
class Units {
 public:
  using Finish = void (*)(void* core, std::shared_ptr<Instruction>& inst);

  Units(uint32_t core_id, uint32_t num_arrays, uint32_t weight_slot_depth);

  // Pick the array of a preload or a matmul; false when a preload finds every weight slot taken.
  bool admit(const std::shared_ptr<Instruction>& inst);
  // The pipeline a typed compute runs on, after admit().
  std::queue<std::shared_ptr<Instruction>>& pipeline(const std::shared_ptr<Instruction>& inst);
  // The weight slot a matmul gives back when it finishes, or null.
  std::shared_ptr<WeightToken> token_to_free(const std::shared_ptr<Instruction>& inst) const;
  // Drop one use of `token`; true when that freed its slot.
  bool release(const std::shared_ptr<WeightToken>& token);
  void count_issued(int compute_type);
  // A barrier over a sparse tag costs the matmuls behind it nothing.
  static void skip_matmuls_behind(const std::shared_ptr<Instruction>& bar);

  // One core cycle of the array pipelines, then of the cross-lane pipeline.
  void cycle(uint64_t core_cycle, void* core, Finish finish);
  bool busy() const;
  int weight_free() const { return _weight_free; }
  uint64_t gemm_inst() const { return _stat_gemm_inst; }
  uint64_t xlu_inst() const { return _stat_xlu_inst; }

  void roll_stats();
  void print_array_stats(uint64_t cycles, bool total) const;
  void print_cross_lane_stats(uint64_t cycles, bool total) const;

 private:
  int pick_free_weight_sa();
  static int matmul_consumers(const std::shared_ptr<Instruction>& preload);

  const uint32_t _core_id;
  const uint32_t _num_arrays;
  uint32_t _array_rr = 0;
  std::vector<std::queue<std::shared_ptr<Instruction>>> _sa_pipeline;
  std::queue<std::shared_ptr<Instruction>> _xlu_pipeline;
  std::vector<int> _weight_slots_used;
  uint32_t _weight_slot_depth = 0;
  int _weight_free = 0;

  std::vector<uint64_t> _stat_sa_cycle, _stat_sa_idle_cycle, _stat_tot_sa_cycle, _stat_tot_sa_idle_cycle;
  uint64_t _stat_xlu_cycle = 0, _stat_xlu_idle_cycle = 0, _stat_tot_xlu_cycle = 0, _stat_tot_xlu_idle_cycle = 0;
  uint64_t _stat_gemm_inst = 0, _stat_xlu_inst = 0;
};

}  // namespace typed
