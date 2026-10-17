#include "TileGraph.h"

int TileSubGraph::_next_id = 0;
TileSubGraph::TileSubGraph() : _ready_tile_queue(), _tile_set(), _id(_next_id++) {
}

void TileSubGraph::add_tile(std::shared_ptr<Tile> tile) {
  for (auto& inst : tile->get_instructions())
    inst->subgraph_id = _id;
  if (tile->get_ready_counter() == 0) {
   _ready_tile_queue.push(tile);
  } else {
    _tile_set.insert(tile);
  }
}

void TileSubGraph::finish_tile(std::shared_ptr<Tile> tile) {
  /* TODO. */
  tile->finish_tile();
  for (auto child_tile_ptr: tile->get_child_tile()) {
    if (child_tile_ptr->get_ready_counter())
      continue;
    /* if child is ready, add ready queue */
    _ready_tile_queue.push(child_tile_ptr);
    _tile_set.erase(child_tile_ptr);
  }
  return;
}

const std::shared_ptr<Tile> TileSubGraph::peek_tile() {
  std::shared_ptr<Tile> ret = std::make_shared<Tile>(Tile::Status::EMPTY);
  if (_ready_tile_queue.empty())
    return ret;
  return _ready_tile_queue.top();
}

std::shared_ptr<Tile> TileSubGraph::get_tile() {
  if (_ready_tile_queue.empty()) {
    std::shared_ptr<Tile> ret = std::make_shared<Tile>(Tile::Status::EMPTY);
    return ret;
  } else {
    std::shared_ptr<Tile> ret = _ready_tile_queue.top();
    _ready_tile_queue.pop();
    return ret;
  }
}


void TileGraph::append_subgraph(std::shared_ptr<TileSubGraph> subgraph) {
  subgraph->init_cache_plan(_cache_plan);
  subgraph->set_progress(_progress);
  _subgraph_vec.push_back(std::move(subgraph));
}

// Materialize the next dispatch tile, on demand. Returns false once the TOG
// is exhausted. Called only when a core has no work left, so the builder never
// runs ahead of the simulation.
bool TileGraph::refill() {
  if (!_pull || _exhausted) return false;
  auto sg = _pull();
  if (!sg) { _exhausted = true; return false; }
  sg->init_cache_plan(_cache_plan);
  sg->set_progress(_progress);
  _subgraph_vec.push_back(std::move(sg));
  return true;
}

// A kernel ends once every work-item has been handed out and finished, and every write of its stores has left the
// interconnect into a memory partition: the next kernel of its stream reads what it wrote.
bool TileGraph::is_finished() {
  if (!_subgraph_vec.empty() || (_pull && !_exhausted) || _progress->tiles || _progress->writes)
    return false;
  for (const auto& core_pair: _cpu_graph_map)
    for (const auto& tile_pair: core_pair.second)
      if (tile_pair.second != nullptr && !tile_pair.second->is_finished())
        return false;
  return true;
}

const std::shared_ptr<Tile> TileGraph::peek_tile(int core_id, int slot_id) {
  std::shared_ptr<Tile> ret = std::make_unique<Tile>(Tile(Tile::Status::EMPTY));
  if (_cpu_graph_map.find(core_id) == _cpu_graph_map.end()) {
    allocate_subgraph(core_id, slot_id);
    return ret;
  } else if (_cpu_graph_map[core_id].find(slot_id) == _cpu_graph_map[core_id].end()) {
    allocate_subgraph(core_id, slot_id);
    return ret;
  } else if (_cpu_graph_map[core_id][slot_id] == nullptr) {
    allocate_subgraph(core_id, slot_id);
    return ret;
  }

  if (_cpu_graph_map[core_id][slot_id]->is_finished()){
    allocate_subgraph(core_id, slot_id);
    return ret;
  }
  return _cpu_graph_map[core_id][slot_id]->peek_tile();
}

std::shared_ptr<Tile> TileGraph::get_tile(int core_id, int slot_id) {
  std::shared_ptr<Tile> ret = std::make_unique<Tile>(Tile(Tile::Status::EMPTY));
  if (_cpu_graph_map.find(core_id) == _cpu_graph_map.end()) {
    allocate_subgraph(core_id, slot_id);
    return ret;
  } else if (_cpu_graph_map[core_id].find(slot_id) == _cpu_graph_map[core_id].end()) {
    allocate_subgraph(core_id, slot_id);
    return ret;
  }

  if (_cpu_graph_map[core_id][slot_id]->is_finished()) {
    allocate_subgraph(core_id, slot_id);
    return ret;
  }
  return _cpu_graph_map[core_id][slot_id]->get_tile();
}

void TileGraph::allocate_subgraph(int core_id, int slot_id) {
  // Drop the finished subgraph: this releases its Instructions.
  if (_cpu_graph_map[core_id][slot_id] != nullptr)
    _cpu_graph_map[core_id][slot_id] = nullptr;

  // Work-items round-robin over the partition's cores, so the next tile for THIS
  // core may be a few dispatches ahead: keep pulling until one turns up.
  for (;;) {
    for (auto it = _subgraph_vec.begin(); it != _subgraph_vec.end(); ++it) {
      if ((*it)->get_core_id() == -1 || (*it)->get_core_id() == core_id) {
        std::shared_ptr<TileSubGraph> subgraph = *it;
        _cpu_graph_map[core_id][slot_id] = subgraph;
        _subgraph_vec.erase(it);
        return;
      }
    }
    if (!refill()) return;      // exhausted: leave the slot empty
  }
}