#include "Result.h"

#include <cmath>
#include <cstdlib>
#include <fstream>

#include <spdlog/spdlog.h>

#include "togsim_runtime.h"

Result& result() {
  static Result r;
  return r;
}

void Result::unit(const std::string& who, const std::string& name, double utilization_pct, int64_t active,
                  int64_t idle) {
  _units[who][name] = Unit{utilization_pct, active, idle};
}

namespace {

// A JSON string: the characters JSON escapes, escaped; a name holds nothing else of note.
std::string quoted(const std::string& s) {
  std::string out = "\"";
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') out += '\\', out += (char)c;
    else if (c < 0x20) out += fmt::format("\\u{:04x}", c);
    else out += (char)c;
  }
  return out + "\"";
}

// A JSON number: a value that is not finite (a utilization over zero cycles) is null.
std::string number(double v) { return std::isfinite(v) ? fmt::format("{}", v) : "null"; }

std::string channel(const Result::Channel& c) {
  return fmt::format("{{\"bandwidth_gbps\": {}, \"utilization_pct\": {}, \"reads\": {}, \"writes\": {}}}",
                     number(c.bandwidth_gbps), number(c.utilization_pct), c.reads, c.writes);
}

template <typename Map, typename Fn>
std::string object(const Map& map, Fn value, const std::string& indent) {
  std::string out = "{";
  const char* sep = "\n";
  for (const auto& [key, v] : map) {
    out += fmt::format("{}{}  {}: {}", sep, indent, quoted(fmt::format("{}", key)), value(v));
    sep = ",\n";
  }
  return out + (map.empty() ? "}" : "\n" + indent + "}");
}

}  // namespace

void Result::write(const std::string& path) const {
  std::string units = object(_units, [](const auto& by_unit) {
    return object(by_unit, [](const Unit& u) {
      return fmt::format("{{\"utilization_pct\": {}, \"active_cycles\": {}, \"idle_cycles\": {}}}",
                         number(u.utilization_pct), u.active_cycles, u.idle_cycles);
    }, "    ");
  }, "  ");
  std::string dma = object(_dma, [](const Dma& d) {
    return fmt::format("{{\"active_cycles\": {}, \"idle_cycles\": {}, \"dram_bw_gbps\": {}, \"responses\": {}}}",
                       d.active_cycles, d.idle_cycles, number(d.dram_bw_gbps), d.responses);
  }, "  ");
  std::string cores = object(_core_cycles, [](uint64_t c) { return fmt::format("{}", c); }, "  ");
  std::string channels = object(_channels, [](const Channel& c) { return channel(c); }, "    ");
  std::string kernels = "[";
  const char* sep = "\n";
  for (const Kernel& k : _kernels) {
    kernels += fmt::format("{}    {{\"scheduler\": {}, \"kernel_id\": {}, \"started\": {}, \"finished\": {}}}", sep,
                           k.scheduler, k.kernel_id, k.started, k.finished);
    sep = ",\n";
  }
  kernels += _kernels.empty() ? "]" : "\n  ]";

  std::ofstream out(path);
  if (!out) {
    spdlog::error("[TOGSim] --result: cannot write {}", path);
    exit(EXIT_FAILURE);
  }
  out << "{\n"
      << fmt::format("  \"version\": {},\n", quoted(TOGSIM_VERSION_STRING))
      << fmt::format("  \"total_cycles\": {},\n", _total_cycles)
      << fmt::format("  \"wall_clock_s\": {},\n", _wall_clock_s < 0 ? "null" : number(_wall_clock_s))
      << "  \"core_cycles\": " << cores << ",\n"
      << "  \"units\": " << units << ",\n"
      << "  \"dma\": " << dma << ",\n"
      << "  \"dram\": {\"channels\": " << channels << ", \"combined\": "
      << (_has_combined ? channel(_combined) : std::string("null")) << "},\n"
      << "  \"kernels\": " << kernels << "\n"
      << "}\n";
  if (!out) {
    spdlog::error("[TOGSim] --result: writing {} failed", path);
    exit(EXIT_FAILURE);
  }
}
