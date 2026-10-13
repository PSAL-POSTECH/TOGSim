# Machine configuration

A machine config is a flat YAML file. `examples/minimal.yml` has every required key and depends on
no other file:

```yaml
num_cores: 1
core_freq_mhz: 1000
core_stats_print_period_cycles: 0

dram_type: simple
dram_channels: 1
dram_latency: 10
dram_stats_print_period_cycles: 0

icnt_type: simple
icnt_freq_mhz: 1000
icnt_latency_cycles: 1

systolic_array_split: none
```

`configs/` ships 19 machines named `systolic_ws_<lanes>x<lanes>_c<cores>_<interconnect>_<machine>`:
`<lanes>` is `vpu_num_lanes`, `<cores>` is `num_cores`, `<interconnect>` is `simple_noc` or
`booksim`, and `<machine>` selects the clock, the arrays per core and the DRAM file. So
`systolic_ws_256x256_c1_simple_noc_tpuv6e.yml`, called tpuv6e in these pages, has 256 lanes, one core and
the simple interconnect. [docs/config-keys.md](config-keys.md) lists every key with its
default, and all 19 configs.

## The knobs

| Question | Keys | Effect |
|---|---|---|
| How many cores, and how are kernels spread over them? | `num_cores`; `num_partition`, `partition: {core_<i>: <id>}` | A kernel's work-items go round-robin over the cores of its partition. All cores are in partition 0 unless `partition` says otherwise. |
| How fast is the core clock? | `core_freq_mhz` | Totals are in cycles of this clock. DRAM and the interconnect tick at their own `dram_freq_mhz` and `icnt_freq_mhz`. |
| How many systolic arrays, how wide? | `num_systolic_array_per_core` (default 1), `vpu_num_lanes` | Used by the array split: how many arrays a compute tile's rows can be spread over, and how many rows one array takes before the compute is split. |
| Is the array split on? | `systolic_array_split`: `round_robin` (default) or `none`; `systolic_array_unit` (default `Systolic`) | With `round_robin` a unit table is required and computes get shorter ([tutorial step 4](tutorial.md#4-two-kinds-of-compute-unit-rows-and-the-array-split)). |
| How large is the scratchpad? | `core_spad_size_kb` (default 0: unlimited) | Limits the buffers alive on a core, and whether one or two work-items are in flight. |
| Which DRAM model? | `dram_type`: `ramulator2` with `ramulator_config_path`, or `simple` with `dram_latency`; `dram_channels` | Ramulator2 models the device its file describes. `simple` answers every request after a fixed latency, optionally under a bandwidth cap. |
| Is there an L2 cache in front of DRAM? | `l2d_type`: `nocache` (default) or `datacache` with `l2d_config` | A TOG marks no address cacheable. The totals of the hello, tiles, gemm and async_barrier examples on tpuv6e are the same with either value. |
| Which interconnect between cores and DRAM? | `icnt_type`: `simple` with `icnt_latency_cycles`, or `booksim2` with `booksim_config_path`; `icnt_injection_ports_per_core` (default 1) | The ports per core are how many DRAM requests a core can send per cycle. |
| How chatty is the log? | `core_stats_print_period_cycles`, `dram_stats_print_period_cycles` | 0 turns the periodic blocks off. They do not change the timing. |

Keys TOGSim does not read are ignored without a message, so check the spelling of a key that seems
to have no effect.

**The array size is not a key.** The `256x256` in a file name matches that config's
`vpu_num_lanes: 256`; TOGSim reads no other array dimension. What a compute costs comes from the
cycle table, and the array reaches the simulator only through `num_systolic_array_per_core`,
`vpu_num_lanes` and the unit table. To model another array, change those keys and supply the two
tables for it.

## Changing the machine

The shipped configs name their Ramulator2 and BookSim2 files relative to themselves
(`../configs/ramulator2_configs/...`), so a copy made elsewhere exits 255. Make that path absolute
while you change a key:

```bash
g++ -std=c++17 -shared -fPIC -O1 -Iinclude examples/tiles/tiles_tog.cpp -o examples/tiles/tog.so
sed -e "s#: \.\./configs/#: $PWD/configs/#" -e 's/^num_cores: .*/num_cores: 2/' \
  configs/systolic_ws_256x256_c1_simple_noc_tpuv6e.yml > build/two_cores.yml
build/bin/Simulator --config build/two_cores.yml --tog_so examples/tiles/tog.so \
  --cycle_table examples/tiles/tog_cycles.tsv --unit_table examples/tiles/tog_units.tsv | tail -n 1
```

```text
[2026-10-12 14:29:13.593] [info] Total execution cycles: 2647
```

The `sed` changes a key the base config has. A key it lacks, such as `systolic_array_split` on
every shipped config, is appended instead: `echo 'systolic_array_split: none' >> build/my.yml`.

**Sweeping a knob** is the same command in a loop. The tiles kernel on one, two and four cores:

```bash
for n in 1 2 4; do
  sed -e "s#: \.\./configs/#: $PWD/configs/#" -e "s/^num_cores: .*/num_cores: $n/" \
    configs/systolic_ws_256x256_c1_simple_noc_tpuv6e.yml > build/cores_$n.yml
  build/bin/Simulator --config build/cores_$n.yml --tog_so examples/tiles/tog.so \
    --cycle_table examples/tiles/tog_cycles.tsv --unit_table examples/tiles/tog_units.tsv \
    | awk -v n=$n '/Total execution cycles/ { print "num_cores", n, $NF }'
done
```

```text
num_cores 1 5032
num_cores 2 2647
num_cores 4 1495
```

Before and after, for the knobs the examples turn (all on tpuv6e, one key changed):

| Example | Key changed | Before | After | What the trace shows |
|---|---|---|---|---|
| tiles (8 work-items) | `num_cores: 1` to `2` | 5032 | 2647 | Four work-items per core. |
| tiles | `num_cores: 1` to `4` | 5032 | 1495 | Two work-items per core. |
| tiles | `core_spad_size_kb: 131072` to `48` | 5032 | 6292 | `max_dispatch=1`: a work-item needs 32768 bytes, more than half the spad, so only one is in flight. |
| tiles | `dram_channels: 32` to `16` | 5032 | 5023 | The first compute starts at 146 instead of 143. |
| gemm (512 rows) | `num_systolic_array_per_core: 2` to `1` | 1106 | 1618 | `compute_cycle=600` instead of 344. |
| gemm (256 rows) | `vpu_num_lanes: 256` to `128` | 1618 | 1362 | `compute_cycle=472` instead of 600. |
| gemm | `systolic_array_split: round_robin` to `none` | 1106 | 1618 | `compute_cycle=600`, and the built-in unit lines. |

`examples/run.sh` makes these copies with its `machine` function and prints one
`# <copy>: <base> with <key>` line for each.
