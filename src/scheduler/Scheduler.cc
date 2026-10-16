#include "Scheduler.h"

Scheduler::Scheduler(SimulationConfig config, const cycle_type* core_cycle, const uint64_t* core_time, int id,
                     std::function<void()> kernel_ended)
    : _id(id), _config(config), _core_cycle(core_cycle), _core_time(core_time),
      _kernel_ended(std::move(kernel_ended)) {
}

void Scheduler::enqueue_graph(std::unique_ptr<TileGraph> tile_graph) {
  _tile_graph.push_back(std::move(tile_graph));
  refresh_status();
}

// Only the front kernel hands out work-items (see get_tile), so the one that finishes is the front kernel's.
void Scheduler::finish_tile(std::shared_ptr<Tile> tile) {
  if (_tile_graph.empty() || &tile->get_owner()->progress() != &_tile_graph.at(0)->progress()) {
    spdlog::error("[Scheduler {}] a work-item finished that the front kernel did not hand out", _id);
    exit(EXIT_FAILURE);
  }
  tile->get_owner()->finish_tile(tile);
  _tile_graph.at(0)->progress().tiles--;
  refresh_status();
}

// The front kernel leaves the queue once it has ended (TileGraph::is_finished).
const std::shared_ptr<Tile> Scheduler::peek_tile(int core_id, int slot_id, CoreType ctype) {
  refresh_status();
  if (_tile_graph.empty() || _tile_graph.at(0)->get_arrival_time() > *_core_cycle)
    return std::make_unique<Tile>(Tile(Tile::Status::EMPTY));
  if (ctype == CoreType::WS_MESH)
    return _tile_graph.at(0)->peek_tile(core_id, slot_id);
  return std::make_unique<Tile>(Tile(Tile::Status::EMPTY));
}

// The front kernel hands out a work-item: it is in flight until finish_tile, and the first one sets the start time.
std::shared_ptr<Tile> Scheduler::get_tile(int core_id, int slot_id) {
  std::shared_ptr<Tile> tile = std::make_unique<Tile>(Tile(Tile::Status::EMPTY));
  if (empty(core_id)) {
    return tile;
  } else {
    tile = std::move(_tile_graph.at(0)->get_tile(core_id, slot_id));
    if (tile->get_status() != Tile::Status::EMPTY) {
      _tile_graph.at(0)->progress().tiles++;
      if (_tile_graph.at(0)->get_start_time() == 0)
        _tile_graph.at(0)->set_start_time(*_core_cycle);
    }
  }
  refresh_status();
  return tile;
}

bool Scheduler::empty() {
  if (_tile_graph.empty())
    return true;
  return false;
}

// A kernel queued behind the front one is work for the core as well.
bool Scheduler::empty(int core_id) {
  for (const auto& graph : _tile_graph)
    if (!graph->empty(core_id)) return false;
  return true;
}

void Scheduler::refresh_status() {
  if (_tile_graph.empty())
    return;

  /* Remove finished request */
  if (_tile_graph.at(0)->is_finished()) {
    unsigned int kernel_id = _tile_graph.at(0)->get_kernel_id();
    cycle_type start_time = _tile_graph.at(0)->get_start_time();
    cycle_type compute_time = 0;
    if (start_time > 0) {
      compute_time = *_core_cycle - start_time;
    } else {
      // Fallback to arrival_time if start_time was not recorded
      start_time = _tile_graph.at(0)->get_arrival_time();
      compute_time = *_core_cycle - start_time;
    }
    
    spdlog::info("[Scheduler {}] Kernel {} has completed - TOG path: {} operation: {} finished at cycle {}",
                 _id, kernel_id, _tile_graph.at(0)->get_graph_path(),
                 _tile_graph.at(0)->get_name(), *_core_cycle);
    spdlog::info("[Scheduler {}] Kernel {} execution summary - Started at: {} cycles, Total compute time: {} cycles",
                 _id, kernel_id, start_time, compute_time);
    _kernel_ended();
    _tile_graph.pop_front();
  }
}