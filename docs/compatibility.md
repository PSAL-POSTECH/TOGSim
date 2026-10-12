# Compatibility

What a compiler, a driver or a script may depend on, and how the version numbers move. The [README](../README.md) explains how to use each of these; this page states the
contract.

## The five things to depend on

A consumer of TOGSim may depend on these five, and on nothing else in this repository:

1. the C ABI of `include/togsim_runtime.h` ([The TOG ABI](../README.md#the-tog-abi));
2. the files beside `tog.so` and the two tables
   ([files](../README.md#files-beside-togso), [tables](../README.md#the-cycle-table));
3. the command line and the exit statuses ([Running TOGSim](../README.md#running-togsim));
4. the output lines listed in [output-lines.md](output-lines.md#lines-a-script-may-rely-on);
5. the machine config keys listed in [config-keys.md](config-keys.md).

The source layout, the classes under `include/` and `src/`, the numbers in any log line and the
lines that list 4 does not name can change in any commit.

## Version numbers

Tags are `vMAJOR.MINOR.PATCH`.

- **MAJOR** moves when 1 or 2 changes incompatibly. `TOGSIM_ABI_VERSION` moves with it, so a TOG
  built against another header is refused by name and version instead of being replayed wrongly.
- **MINOR** moves when 3, 4 or 5 changes.
- **PATCH** changes no contract.

`TOGSIM_ABI_VERSION` is 16. Including the header defines the symbol `togsim_abi_version` in the
TOG with that value, and TOGSim compares it when it loads the TOG.

## Using TOGSim from another repository

A consumer pins this repository as a git submodule by commit, builds it from source, and names it
with one environment variable, `TOGSIM_DIR`: the header is `$TOGSIM_DIR/include/togsim_runtime.h`
and the binary `$TOGSIM_DIR/build/bin/Simulator`. TOGSim itself reads no environment variable.

The binary finds `build/lib/libramulator.so` by an absolute path recorded at link time. Run it from
the build tree it was built in; a build tree moved to another path does not start.

## Deprecated: `compute_type` 1, 2 and 3

Pass `compute_type` 0 to `togsim_compute` and say which units a compute used in the unit table.
The values 1, 2 and 3 name the unit in the call instead. They still replay, with one warning when
the TOG is loaded, and will be removed together with the config key `sa_weight_buffer_depth`.
