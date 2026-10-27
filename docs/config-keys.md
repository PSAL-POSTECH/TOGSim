# Machine config keys

The machine config is the YAML file given to `--config`: a flat map of the keys below. [machine.md](machine.md) groups the keys by the question they answer and shows
what changing one does; this page is the full list.

- A required key that is missing, or a value TOGSim does not implement, aborts the run with exit
  status 134 and a `what():  <reason>` line on stderr.
- A key that is not in this list is named in a `[warning]` line and otherwise ignored, unless another tool
  reads it: `isa`, `vcix`, `address_map` (the compiler; the frontend takes the vector length from the
  `zvl<N>b` entry of `isa`).
- The config is echoed at the top of the log, so a run shows which values it used.

## Cores

| Key | Required | Default | Meaning |
|---|---|---|---|
| `num_cores` | yes | | The number of cores. |
| `core_freq_mhz` | yes | | The core clock. `Total execution cycles` counts cycles of this clock. |
| `core_stats_print_period_cycles` | yes | | Print a `Core stat` block every this many core cycles; 0 prints none. |
| `core_type` | no | all `ws_mesh` | A list with one entry per core. `ws_mesh` is the only type. |
| `num_systolic_array_per_core` | no | 1 | The arrays a tile's rows can be spread over by the array split. |
| `vpu_num_lanes` | unless `systolic_array_split: none` | | The rows one array takes: a tile with more rows than this is split. |
| `systolic_array_split` | no | `round_robin` | `round_robin` spreads a tile's array rows over the arrays and needs a unit table. `none` turns that off. |
| `systolic_array_unit` | unless `systolic_array_split: none` | | The array's component (`TPUV6E_MXU`): its unit in the unit table is that name and `.rows` (`TPUV6E_MXU.rows`). Missing with the split, or set with `systolic_array_split: none`, it aborts. |
| `vpu_spad_size_kb_per_lane` | no | | The scratchpad of one lane, in KB, what a dispatch's [`.spad`](tog-abi.md#buffers-and-the-scratchpad) (bytes per lane) is measured against: a dispatch that states more is refused, and a core keeps two work-items in flight only when twice the `.spad` fits. Without it (or at 0) the spad is unlimited. |
| `core_spad_size_kb` | no | | The scratchpad of one core, in KB. It may only restate `vpu_spad_size_kb_per_lane` x `vpu_num_lanes`: another value aborts, and so does `core_spad_size_kb` without `vpu_spad_size_kb_per_lane`, since a dispatch states its spad per lane. |
| `sa_weight_buffer_depth` | no | 2 | Deprecated: read only for a TOG that passes `compute_type` 1, 2 or 3. |
| `dma_load_latency_cycles` | no | 0 | Core cycles the DMA engine holds a load before its first request. The engine sends one DMA at a time, so the loads pay it one after another, asynchronous ones too. |
| `dma_store_latency_cycles` | no | 0 | Core cycles the DMA engine holds a store before its first request, as `dma_load_latency_cycles` for a load. |
| `dma_load_ready_cycles` | no | 0 | Core cycles from a load's last data to its completion, waited and not held: the engine goes on to the next DMA, so loads in flight together wait theirs at the same time and pay it once, and a load that waits on another pays it again. The v6e's measured DMA cost has this shape (tpu-perf-archive `data/v6e/dma`: about 0.57 us a load in a chain, 18 ns more for each load in parallel). |
| `dma_store_ready_cycles` | no | 0 | Core cycles a store keeps its core running after its requests are out. The store's work-item, its dispatch slot and the next kernel go on at once; only the end of the run, or a `DEVICE_SYNC`, waits for it. |

## DRAM

| Key | Required | Default | Meaning |
|---|---|---|---|
| `dram_type` | yes | | `ramulator2` or `simple`. |
| `dram_channels` | yes | | The number of DRAM channels. |
| `dram_stats_print_period_cycles` | no | none: set it | Print a `[DRAM] all N channels combined` line every this many cycles; 0 prints none. |
| `dram_num_partitions` | no | 1 | Splits the channels into that many equal groups. A request to a channel of the group numbered like the core's partition counts as `NUMA local`, any other as `remote`. Must divide `dram_channels`. |
| `ramulator_config_path` | with `ramulator2` | | The Ramulator2 file. A relative path is relative to the directory of the config file. |
| `dram_freq_mhz` | no | see right | With `ramulator2` the DRAM clock comes from the Ramulator2 file, and a stated value that differs aborts. With `simple` it defaults to `core_freq_mhz`. |
| `dram_latency` | with `simple` | | Cycles from a request to its response. |
| `dram_req_size_byte` | no (`simple` only) | 32 | Bytes per DRAM request. With `ramulator2` the Ramulator2 file decides. |
| `dram_bandwidth_gbps_per_channel` or `dram_bandwidth_gbps_total` | no (`simple` only) | no cap | A bandwidth cap. Give at most one of the two, together with `dram_freq_mhz`. |

## L2 cache

| Key | Required | Default | Meaning |
|---|---|---|---|
| `l2d_type` | no | `nocache` | `nocache` or `datacache`. A TOG marks no address cacheable (`cacheable=false` on every `L2CACHE_STAT` trace line), and the totals of the hello, tiles, gemm and async_barrier examples on tpuv6e are the same with either value. |
| `l2d_config` | with `datacache` | | The cache geometry, as a string in the form the shipped tpuv4 and tpuv6e configs use. |
| `l2d_hit_latency` | no | 1 | Cycles of a hit. |

## Interconnect

| Key | Required | Default | Meaning |
|---|---|---|---|
| `icnt_type` | yes | | `simple` or `booksim2`. |
| `icnt_freq_mhz` | yes | | The interconnect clock. |
| `icnt_latency_cycles` | with `simple`: set it | none | Cycles a request spends in the interconnect. |
| `booksim_config_path` | with `booksim2` | | The BookSim2 file. A relative path is relative to the directory of the config file. |
| `icnt_injection_ports_per_core` | no | 1 | How many DRAM requests a core can send per cycle. |
| `icnt_stats_print_period_cycles` | no | 0 | Print interconnect statistics every this many cycles; 0 prints none. |

## Partitions

| Key | Required | Default | Meaning |
|---|---|---|---|
| `num_partition` | no | 1 | The number of partitions. A kernel runs on the cores of one partition. |
| `partition` | no | every core in 0 | A map `core_<i>: <partition id>`. A core it leaves out goes to partition 0, with a warning. |

`--tog_so` runs its TOG on partition 0. In a command stream the `stream_index` field of
`LAUNCH_KERNEL` is the partition.

## TOG layout

| Key | Required | Default | Meaning |
|---|---|---|---|
| `tog_tensor_tile` | no | | `[rows, cols]`: lay every tensor argument out in DRAM in tiles of that many 4-byte elements, for timing only. Needs `tog_tensor_shapes.txt` beside `tog.so`. |

## Read by the compiler, not by TOGSim

| Key | Required | Default | Meaning |
|---|---|---|---|
| `isa` | no | | The ISA the compiler targets, ending in the accelerator `pytorchsim_<model>`. Read by the compiler, not by TOGSim. |
| `vcix` | no | | The vcix model's own keys, passed to it as they are. Read by the compiler, not by TOGSim. |
| `address_map` | no | | Where the spad and DRAM sit in the address space. Read by the compiler, not by TOGSim. |

## The shipped configs

`configs/` holds 7 machines named `<machine>[_<n>x<n>][_booksim][_<variant>]`:
`<machine>` selects the clock, the arrays per core and the Ramulator2 file; `_<n>x<n>` marks the
8- and 32-lane configs, and `_booksim` `icnt_type: booksim2` (else `simple`). Each has one core; a test
that needs more sets `num_cores` (and `partition`) over one of them. All 7 use
`dram_type: ramulator2`, and none sets `systolic_array_split: none`, so every one needs a unit table.

| Config (without `.yml`) | Cores | Core MHz | Arrays per core | Lanes | Spad KB | DRAM channels | Ramulator2 file | L2 | Interconnect |
|---|---|---|---|---|---|---|---|---|---|
| `tpuv3_booksim` | 1 | 940 | 2 | 128 | 16384 | 16 | `HBM2_TPUv3.yaml` | none | BookSim2 `fly_c16_m16.icnt` |
| `tpuv3` | 1 | 940 | 2 | 128 | 16384 | 16 | `HBM2_TPUv3.yaml` | none | simple, 10 cycles |
| `tpuv3_half` | 1 | 940 | 2 | 128 | 16384 | 8 | `HBM2_TPUv3.yaml` | none | simple, 10 cycles |
| `tpuv4` | 1 | 1050 | 4 | 128 | 16384 | 16 | `HBM2_TPUv4.yaml` | data cache | simple, 10 cycles |
| `tpuv6e` | 1 | 1751 | 2 | 128 | 131072 | 32 | `HBM3_TPUv6e.yaml` | data cache | simple, 10 cycles |
| `tpuv3_32x32` | 1 | 940 | 2 | 32 | 1024 | 16 | `HBM2_TPUv3.yaml` | none | simple, 10 cycles |
| `tpuv3_8x8` | 1 | 940 | 2 | 8 | 256 | 16 | `HBM2_TPUv3.yaml` | none | simple, 10 cycles |

- `tpuv6e` is the machine the tests and the examples use: one TPU v6e
  TensorCore as libtpu drives it, two 256x256 arrays, 128 lanes of 1024 KB (VMEM 128 MB) and 256-bit
  vectors (8 sublanes x 32 bit), and `systolic_array_split: round_robin` over the array unit
  `TPUV6E_MXU.rows` (`systolic_array_unit: TPUV6E_MXU`). It prints statistics every 200 cycles; the others every 10000.
- Every shipped config states `isa`, `vcix` and `address_map` and names its array's component: `TPUV3_MXU` on the
  `tpuv3` ones, `TPUV4_MXU` on the `tpuv4` ones. `tpuv3` states its model's (Coprocessor `libtpuv3.so`) keys,
  each tpuv6e's value, none measured on a v3 (its comments say which); the other `tpuv3` ones and `tpuv4` (no
  `pytorchsim_tpuv4` model yet) have `vcix: {}`.
- `tpuv3_half` is `tpuv3` with half the DRAM channels.
- `examples/minimal.yml` is not one of the 7: it is a `dram_type: simple` machine that needs no
  other file and no unit table.

Every shipped config writes its Ramulator2 and BookSim2 paths as `../configs/...`, which resolves
only while the file sits in a directory beside `configs/` or in `configs/` itself. [machine.md](machine.md#changing-the-machine) shows how to copy one.
