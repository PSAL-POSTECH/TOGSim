#pragma once

#include <spdlog/fmt/ranges.h>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <cassert>
#include <cstdint>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <string>
#include <vector>

#include "SimulationConfig.h"
#include "Instruction.h"

#define MIN(x, y) (((x) > (y)) ? (y) : (x))
#define MIN3(x, y, z) MIN(MIN(x, y), z)
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

#define KB *1024

#define PAGE_SIZE 4096

typedef uint64_t addr_type;
typedef uint64_t cycle_type;

bool loadConfig(const std::string& config_path, YAML::Node& config_yaml);

// The config as TOGSim reads it: every key looked up goes through operator[], which notes it, so that the keys of
// the file nothing looked up can be named.
class ConfigReader {
 public:
  explicit ConfigReader(const YAML::Node& yaml) : _yaml(yaml) {}
  YAML::Node operator[](const std::string& key) {
    _read.insert(key);
    return _yaml[key];
  }
  const std::set<std::string>& read() const { return _read; }

 private:
  const YAML::Node& _yaml;
  std::set<std::string> _read;
};

SimulationConfig initialize_config(const YAML::Node& config,
                                     const std::string& config_file_path = {});