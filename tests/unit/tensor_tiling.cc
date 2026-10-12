// One (8,128) fp32 tile of a 512x512 tensor, loaded by a MOVIN: row-major it is eight 512 B runs at a 2 KiB
// pitch; in (8,128) tiles it is the tensor's first 4 KiB, contiguous. tensor_tiling_test [check] runs the
// named check, or all of them: exit 1 if one failed, 2 if no check has that name.
#include <cstdio>
#include <cstring>
#include <memory>
#include <set>
#include <vector>

#include "Instruction.h"

static std::set<addr_type> requests(std::shared_ptr<const TensorTiling> tiling) {
  Instruction inst(Opcode::MOVIN, 0, 0, 0x10000000, {8, 128}, {512, 1}, 32, {0}, {1}, {});
  inst.set_tensor_tiling(std::move(tiling));
  return *inst.get_dram_address(32);
}

static bool row_major_is_pitched_runs() {
  std::set<addr_type> pitched;
  for (addr_type r = 0; r < 8; ++r)
    for (addr_type a = 0; a < 512; a += 32) pitched.insert(0x10000000 + r * 2048 + a);
  return requests(nullptr) == pitched;
}

static bool tiled_is_one_run() {
  std::set<addr_type> run;
  for (addr_type a = 0; a < 4096; a += 32) run.insert(0x10000000 + a);
  return requests(std::make_shared<TensorTiling>(TensorTiling{0x10000000, 512, 512, 4, 8, 128})) == run;
}

static bool footprint_pads_to_whole_tiles() {
  return TensorTiling{0, 9, 130, 2, 16, 128}.footprint() == 16 * 256 * 2;
}

int main(int argc, char** argv) {
  const struct { const char* name; const char* what; bool (*holds)(); } checks[] = {
      {"row_major_is_pitched_runs", "row-major: eight 512 B runs at a 2 KiB pitch", row_major_is_pitched_runs},
      {"tiled_is_one_run", "(8,128) tiles: one contiguous 4 KiB run", tiled_is_one_run},
      {"footprint_pads_to_whole_tiles", "footprint pads to whole tiles", footprint_pads_to_whole_tiles},
  };
  int ran = 0, failed = 0;
  for (const auto& check : checks) {
    if (argc > 1 && std::strcmp(argv[1], check.name) != 0) continue;
    const bool ok = check.holds();
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", check.what);
    ran++;
    failed += !ok;
  }
  if (!ran) {
    std::fprintf(stderr, "no check named %s\n", argv[1]);
    return 2;
  }
  return failed ? 1 : 0;
}
