#pragma once
// UnitTable -- the per-tile unit admission table (--unit_table): what each unit admitted, at
// its primary port, during ONE run of a compute tile. Core sums it per completion and prints
// each unit as utilized cycles = admitted / capacity_per_cycle over its cycles.
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

class UnitTable {
 public:
  struct Unit {
    std::string name;
    std::string unit_of_work;
    uint64_t capacity;
  };

  // Parse and validate `path`; on any error, log the offending line and exit(1).
  static std::unique_ptr<UnitTable> load(const std::string& path);
  // The rows of another TOG's table `path`, its tile_ids shifted behind the ones held: returns
  // that shift. A unit the table does not have yet is an error (one machine, one set of units).
  size_t append(const std::string& path) { return read(path, false); }

  size_t num_units() const { return _units.size(); }
  const std::string& unit_name(size_t u) const { return _units[u].name; }
  uint64_t capacity(size_t u) const { return _units[u].capacity; }
  // The unit named `name`, or -1.
  int64_t find(const std::string& name) const {
    for (size_t u = 0; u < _units.size(); u++) if (_units[u].name == name) return (int64_t)u;
    return -1;
  }
  // systolic_array_split: unit `u` is the array, a compute spreads its rows over up to `arrays` of them, one per
  // `lanes` rows. Not called when the split is off: array_unit() stays -1.
  void set_array_unit(size_t u, uint64_t lanes, uint64_t arrays) {
    _array_unit = (int64_t)u;
    _lanes = lanes;
    _arrays = arrays;
  }
  int64_t array_unit() const { return _array_unit; }
  struct Window {
    uint64_t cycles;   // what one run of the tile takes
    uint32_t n_eff;    // the arrays its array-unit rows are spread over
  };
  // B_sys = array-unit rows / capacity, N_eff = min(arrays, max(1, ceil(rows / lanes))); the window keeps
  // cc - B_sys and runs the longest of B_sys / N_eff and every other unit's B_u, never longer than `cc`.
  // Without the split, with N_eff 1 or with cc 0: cc as is, on one array.
  Window window(int64_t tile_id, uint64_t cc) const;
  // The first unit one run of `tile_id` asks more of than the unit takes in that window (admitted > capacity x
  // cycles, x N_eff for the array unit), as the clause a refusal ends with; empty when every unit's work fits.
  std::string overfull(int64_t tile_id, uint64_t cc) const;
  // The (unit, admitted) rows of one run of `tile_id`; empty for an unknown tile.
  const std::vector<std::pair<size_t, uint64_t>>& rows(int64_t tile_id) const {
    static const std::vector<std::pair<size_t, uint64_t>> none;
    return has_rows(tile_id) ? _rows[tile_id] : none;
  }
  // Add one run of `tile_id` into `acc` (indexed like the units); unknown tiles add nothing.
  void accumulate(int64_t tile_id, std::vector<uint64_t>& acc) const {
    if (tile_id < 0 || (size_t)tile_id >= _rows.size()) return;
    for (auto& [u, admitted] : _rows[tile_id]) acc[u] += admitted;
  }
  // True if `tile_id` has at least one row.
  bool has_rows(int64_t tile_id) const {
    return tile_id >= 0 && (size_t)tile_id < _rows.size() && !_rows[tile_id].empty();
  }
  // Add `fraction` of one run of `tile_id` into `acc`, the array unit's rows divided by the run's
  // `n_eff` arrays: per-array credit, checked against one array's capacity.
  void accumulate(int64_t tile_id, double fraction, std::vector<double>& acc, uint32_t n_eff = 1) const {
    if (!has_rows(tile_id)) return;
    for (auto& [u, admitted] : _rows[tile_id])
      acc[u] += fraction * (double)admitted / ((int64_t)u == _array_unit ? (double)n_eff : 1.0);
  }
  // Print '<who> : <unit> utilization(%): ..., active_cycles: U, idle_cycles: I' for unit `u`;
  // false if `check` and the unit admitted more than its capacity allows in `cycles`.
  bool print(const std::string& who, size_t u, uint64_t admitted, uint64_t cycles, bool check) const;
  // The same line for a fractional (time-spread) admission; idle never goes negative.
  // False if the line exceeds 100%.
  bool print_spread(const std::string& who, size_t u, double admitted, uint64_t cycles) const;

 private:
  size_t read(const std::string& path, bool new_units);
  std::vector<Unit> _units;   // in first-seen order
  int64_t _array_unit = -1;
  uint64_t _lanes = 0, _arrays = 1;   // of the array split
  std::vector<std::vector<std::pair<size_t, uint64_t>>> _rows;   // tile_id -> (unit, admitted)
};
