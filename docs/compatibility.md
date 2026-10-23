# Compatibility

What a compiler, a driver or a script may depend on, and how the version numbers move.

## The five things to depend on

A consumer of TOGSim may depend on these five, and on nothing else in this repository:

1. `include/togsim.hpp` ([the API](tog-api.md)) and the [C ABI](tog-abi.md) under it, of one version;
2. the files beside `tog.so`, the two tables and the runtime attribute records
   ([files](tog-abi.md#files-beside-togso), [tables](tog-abi.md#the-cycle-table),
   [records](tog-abi.md#runtime-attributes));
3. the command line and the exit statuses ([Running TOGSim](running.md#running-togsim));
4. the output lines listed in [output-lines.md](output-lines.md#lines-a-script-may-rely-on);
5. the machine config keys listed in [config-keys.md](config-keys.md).

The source layout, the classes under `include/` and `src/`, the numbers in any log line and the
lines that list 4 does not name can change in any commit.

## Version numbers

TOGSim has one version, `MAJOR.MINOR.PATCH`, today 1.0.0, and tags are `vMAJOR.MINOR.PATCH`. The
API of togsim.hpp and the C ABI under it have no number of their own: `TOGSIM_VERSION_MAJOR`,
`_MINOR` and `_PATCH` in `include/togsim_runtime.h` are its only definition.

- **MAJOR** moves when 1 or 2 changes incompatibly.
- **MINOR** moves when 3, 4 or 5 changes.
- **PATCH** changes no contract.

A TOG embeds the version it was built against in `togsim_abi_version`, and a runtime attribute
file the version it was recorded at, both as `MAJOR << 16 | MINOR << 8 | PATCH` (`TOGSIM_VERSION`,
0x10000 for 1.0.0). TOGSim accepts its own MAJOR, no newer than itself, and refuses anything else
at load, naming both versions, instead of simulating it wrongly. The encoding is at least 0x10000,
so the integer ABIs 1 to 17 of before 1.0.0 never pass for it: a TOG built against one is refused
as `built against TOG ABI 16, TOGSim is version 1.0.0` ([what changed](tog-api.md#from-the-integer-abis-to-100)).

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
