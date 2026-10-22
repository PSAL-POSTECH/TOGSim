# Compatibility

What a compiler, a driver or a script may depend on, and how the version numbers move.

## The five things to depend on

A consumer of TOGSim may depend on these five, and on nothing else in this repository:

1. `include/togsim.hpp` ([API v1.0.0](tog-api.md)) and the [C ABI 16](tog-abi.md) under it;
2. the files beside `tog.so` and the two tables
   ([files](tog-abi.md#files-beside-togso), [tables](tog-abi.md#the-cycle-table));
3. the command line and the exit statuses ([Running TOGSim](running.md#running-togsim));
4. the output lines listed in [output-lines.md](output-lines.md#lines-a-script-may-rely-on);
5. the machine config keys listed in [config-keys.md](config-keys.md).

The source layout, the classes under `include/` and `src/`, the numbers in any log line and the
lines that list 4 does not name can change in any commit.

## Version numbers

Tags are `vMAJOR.MINOR.PATCH`.

- **MAJOR** moves when 1 or 2 changes incompatibly. `TOGSIM_ABI_VERSION` (16) moves with an
  incompatible C ABI: TOGSim compares it with the TOG's `togsim_abi_version` at load and refuses a
  TOG built against another, instead of simulating it wrongly.
- **MINOR** moves when 3, 4 or 5 changes.
- **PATCH** changes no contract.

## Using TOGSim from another repository

A consumer pins this repository as a git submodule by commit, builds it from source, and names it
with one environment variable, `TOGSIM_DIR`: the header is `$TOGSIM_DIR/include/togsim.hpp`
and the binary `$TOGSIM_DIR/build/bin/Simulator`. TOGSim itself reads no environment variable.

The binary finds `build/lib/libramulator.so` by an absolute path recorded at link time. Run it from
the build tree it was built in; a build tree moved to another path does not start.

## Deprecated: `compute_type` 1, 2 and 3

togsim.hpp always passes `compute_type` 0; the unit table says which units a compute used. The
values 1, 2 and 3 (raw C ABI only) name the unit in the call instead; they still run, with one
warning at load, and will be removed together with the config key `sa_weight_buffer_depth`.
