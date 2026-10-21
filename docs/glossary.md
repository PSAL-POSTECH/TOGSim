# Glossary

| Term | Meaning |
|---|---|
| tile-operation-graph (TOG) | A shared object (`tog.so`) whose calls are one kernel's DMAs, computes and barriers. It holds control flow only. |
| kernel | One operation to simulate: one TOG with one entry point, its `entry` (`togsim_kernel` in the C ABI). |
| replay | TOGSim running a TOG's calls on the modeled machine. |
| work-item | One `dispatch` (`togsim_dispatch`): a body function plus its arguments, bound to one core. The unit of parallelism. The log calls it a tile. |
| body | The function of a work-item; its calls are the work-item's instructions. |
| instruction | One `load` or `store` (`togsim_dma`), `compute` (`togsim_compute`) or `wait` (`togsim_memory_barrier`) of a body. |
| compute tile, `tile_id` | A kind of compute: a row of the cycle table and the key of its unit-table rows. A compute names it as `node.id + node.count * variant`. |
| data tile | The block of a tensor that one DMA moves, given by `offset`, `dims` and `strides`. |
| tensor argument | An array of the kernel that lives in DRAM. A TOG names it by `Desc.ptr` (`arg_id` in the C ABI); `tog_tensors.txt` gives its size. |
| DMA | A transfer between DRAM and a core's scratchpad: a load (`MOVIN` in the log) or a store (`MOVOUT`). |
| scratchpad (spad), buffer | The memory of a core that DMAs fill and drain. A TOG names its regions, the buffers, by small integer ids. |
| batch, semaphore, barrier | Asynchronous loads join a batch (a semaphore in the C ABI); a `wait`, a barrier (`MEMORY_BAR`) in the log, waits for the data of its loads. |
| cycle table | `tile_id` to cycles: what each compute tile costs. You supply it. |
| unit table | What each hardware unit (the systolic array, the vector unit, ...) took in during one run of each compute tile. You supply it. |
| unit, unit of work, admitted | A hardware unit named in the unit table; what it counts (rows, cycles); how many of those it took in. |
| lanes | `vpu_num_lanes`: the rows one systolic array takes before a compute tile is split over several arrays. |
| systolic array split | Spreading a compute tile's array work over the arrays of a core, which shortens the compute. On by default. |
| machine config | The YAML file given to `--config`. tpuv6e is `configs/tpuv6e.yml`. |
| partition | A set of cores with its own queue of kernels. A kernel runs on the cores of one partition. |
| session, command stream | A run of several kernels, launched by the lines of the file given to `--models_list`. |
| trace | The per-instruction log of a replay, printed at `--log_level trace`. |
| `max_dispatch` | On a `TILE_SCHEDULED` line: how many work-items that core keeps in flight, 1 or 2. |
