#include "Instruction.h"

#include <utility>

#include <fmt/format.h>

uint64_t Instruction::_next_global_inst_id = 0;

std::string format_tag_key_list_hex(const std::vector<int64_t>& tag_keys) {
  if (tag_keys.empty()) {
    return {};
  }
  std::string out;
  for (size_t i = 0; i < tag_keys.size(); ++i) {
    if (i > 0) {
      out.push_back(',');
    }
    out += fmt::format("0x{:016x}", static_cast<uint64_t>(tag_keys[i]));
  }
  return out;
}

std::string opcode_to_string(Opcode opcode) {
    switch (opcode) {
        case Opcode::MOVIN:        return "MOVIN";
        case Opcode::MOVOUT:       return "MOVOUT";
        case Opcode::COMP:         return "COMP";
        case Opcode::MEMORY_BAR:   return "MEMORY_BAR";
        default:                   return "Unknown";
    }
}

Instruction::Instruction(Opcode opcode, cycle_type compute_cycle, size_t num_parents,
            addr_type dram_addr, std::vector<size_t> tile_size, std::vector<int> tile_stride, size_t elem_bits,
            std::vector<int64_t> tag_idx_list, std::vector<int64_t> tag_stride_list,
            std::vector<int64_t> accum_tag_idx_list)
  // The vectors are taken by value, so move them into the members: copying them
  // allocated a second buffer per vector for every DMA and barrier instruction.
  : opcode(opcode), compute_cycle(compute_cycle), ready_counter(num_parents), dram_addr(dram_addr),
    tile_size(std::move(tile_size)), tile_stride(std::move(tile_stride)), _elem_bits(elem_bits),
    _tag_idx_list(std::move(tag_idx_list)), _tag_stride_list(std::move(tag_stride_list)),
    _accum_tag_idx_list(std::move(accum_tag_idx_list)) {
  _global_inst_id = _next_global_inst_id++;
  assert(_tag_idx_list.size()==_tag_stride_list.size());
  _tile_numel = 1;
  for (auto dim : this->tile_size)   // the parameter was moved from
    _tile_numel *= dim;
}

Instruction::Instruction(Opcode opcode)
  : opcode(opcode) {
  _global_inst_id = _next_global_inst_id++;
  _tile_numel = 1;
}

bool DepLess::operator()(const std::shared_ptr<Instruction>& a,
                         const std::shared_ptr<Instruction>& b) const {
  return a->get_global_inst_id() < b->get_global_inst_id();
}

void Instruction::finish_instruction() {
  fire(DepEvent::DONE);   // latency consumers
  finished = true;
}

void Instruction::inc_waiting_request() {
  _nr_waiting_request++;
}

void Instruction::dec_waiting_request() {
  assert(_nr_waiting_request!=0);
  _nr_waiting_request--;
}

void Instruction::prepare_tag_key() {
  /* Calculate tag key */
  int64_t key_offset = 0;
  // exact size: the two unconditional pushes otherwise grow 0 -> 1 -> 2, i.e. an
  // allocate + reallocate + free for every DMA and barrier.
  _tag_key.reserve(2 + _accum_tag_idx_list.size());
  _tag_key.push_back(_addr_id);
  for (size_t i = 0; i < _tag_idx_list.size(); i++)
    key_offset += _tag_idx_list.at(i) * _tag_stride_list.at(i);
  for (auto accum_dim : _accum_tag_idx_list)
    _tag_key.push_back(accum_dim);
  _tag_key.push_back(key_offset);
}

void Instruction::print() {
  spdlog::info("{}", opcode_to_string(opcode));
}

// The dram_req_size-aligned requests of the tile, row-major over its (up to 4) dims. A gather or
// scatter adds each element's element offset (set_element_offsets) to its address; a tiled tensor
// (set_tensor_tiling) then moves each element into its tile before the alignment.
std::shared_ptr<std::set<addr_type>> Instruction::get_dram_address(addr_type dram_req_size) {
  auto address_set = std::make_shared<std::set<addr_type>>();
  size_t index_count = 0;
  while (tile_size.size() < 4)
    tile_size.insert(tile_size.begin(), 1);

  while (tile_stride.size() < 4)
    tile_stride.insert(tile_stride.begin(), 0);

  for (int dim0=0; dim0<tile_size.at(0); dim0++) {
    for (int dim1=0; dim1<tile_size.at(1); dim1++) {
      for (int dim2=0; dim2<tile_size.at(2); dim2++) {
        for (int dim3=0; dim3<tile_size.at(3); dim3++) {
          addr_type address = dim0*tile_stride.at(tile_stride.size() - 4) + \
                              dim1*tile_stride.at(tile_stride.size() - 3) + \
                              dim2*tile_stride.at(tile_stride.size() - 2) + \
                              dim3*tile_stride.at(tile_stride.size() - 1);
          address = dram_addr + ((address * _elem_bits + 7) >> 3);
          if (!_element_offsets.empty())
            address += (addr_type)(_element_offsets[index_count++] * (int64_t)_elem_bits / 8);
          if (_tiling) address = _tiling->map(address);
          address_set->insert(address - (address & dram_req_size-1));
        }
      }
    }
  }
  return address_set;
}
