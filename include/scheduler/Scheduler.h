#pragma once
#include "Tile.h"
#include "Common.h"
#include "TileGraph.h"
#include "SimulationConfig.h"
#include <functional>

class Scheduler {
 public:
  // `kernel_ended` is called as the front kernel, ended, leaves the queue.
  Scheduler(SimulationConfig config, const cycle_type* core_cycle, const uint64_t* core_time, int id,
            std::function<void()> kernel_ended);
  void enqueue_graph(std::unique_ptr<TileGraph> tile_graph);
  void finish_tile(std::shared_ptr<Tile> tile);

  /* For other schedulers */
  virtual std::shared_ptr<Tile> get_tile(int core_id=0, int slot_id=0);
  virtual const std::shared_ptr<Tile> peek_tile(int core_id=0, int slot_id=0, CoreType ctype=CoreType::WS_MESH);
  virtual bool empty();
  virtual bool empty(int core_id);
  virtual void refresh_status();

 protected:
  int _id;
  const cycle_type* _core_cycle;
  const uint64_t* _core_time;
  std::deque<std::unique_ptr<TileGraph>> _tile_graph;
  SimulationConfig _config;
  std::function<void()> _kernel_ended;

  struct CompareTile {
    bool operator()(const std::shared_ptr<Tile>& a, const std::shared_ptr<Tile>& b) const {
      if (a->get_ready_counter() == b->get_ready_counter()) {
        return a->get_required_sram_size() > b->get_required_sram_size();
      }
      return a->get_ready_counter() > b->get_ready_counter();
    }
  };
};