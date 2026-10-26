"""Kernels of one stream run in order: a kernel ends once its writes are in memory, and the next one reads after."""
import re

from togtest import MIB, Case, write_table

KERNELS, ITEMS, REQUESTS = 3, 3, 512
COMPLETED = re.compile(r"\[Scheduler 0\] Kernel (\d+) has completed .* finished at cycle (\d+)")
SYNCED = "Device synchronization completed"
TO_MEMORY = re.compile(r"\[(\d+)\]\[Core \d+\]\[REQ_TO_MEM *\] (MOVIN|MOVOUT) dram=0x([0-9a-f]+) channel=\d+$", re.M)
DRAM = re.compile(r"\[(REQ_TO_DRAM|DRAM_WRITE_ACK) *\] (MOVIN|MOVOUT) dram=0x([0-9a-f]+) channel=(\d+)$", re.M)


def launch(t, kernels, cores=1, sync_between=False, defines=()):
    """--models_list: kernels 0 to `kernels` - 1 of the chain built with `defines`, on partition 0, a DEVICE_SYNC
    after the last and, with `sync_between`, after each."""
    lines = []
    for k in range(kernels):
        home = t.tog("chain_tog.cpp", (f"IN={k}", *defines), tensors=(MIB,) * (kernels + 1), into=f"k{k}").parent
        write_table(home / "tog_cycles.tsv", [(600, 0)])
        write_table(home / "tog_units.tsv", [(0, "TPUV6E_MXU.rows", "rows", 1, 256), (0, "Vpu", "cycles", 1, 100)])
        lines.append(f"LAUNCH_KERNEL,{k},0,0,{home}/kernel,{home}/attributes,0\n")
        if sync_between or k == kernels - 1:
            lines.append("DEVICE_SYNC,0,0,0,-,-,0\n")
    stream = t.work_dir / "commands.txt"
    stream.write_text("".join(lines))
    return t.execute("Simulator", ["--config", t.machine(num_cores=cores), "--models_list", stream,
                                   "--log_level", "trace"]).expect_ok()


def chain(cores):
    """Kernels 0, 1, 2 launched with no DEVICE_SYNC between on `cores` cores, kernel k loading tensor k and storing
    tensor k + 1 (each 1 MiB, at k MiB), which kernel k + 1 loads, each work-item what another stored. A load of an
    address enters the memory partition after the store to it, and the DRAM model after the store's write is acked;
    kernel k completes no earlier than its last store."""
    def case(t):
        run = launch(t, KERNELS, cores=cores)
        requests = [(int(m[1]), m[2], int(m[3], 16)) for m in TO_MEMORY.finditer(run.log)]
        stored = {addr: cycle for cycle, opcode, addr in requests if opcode == "MOVOUT"}
        early = [(addr, cycle, stored[addr]) for cycle, opcode, addr in requests
                 if opcode == "MOVIN" and addr in stored and cycle <= stored[addr]]
        completed = {int(m[1]): int(m[2]) for m in COMPLETED.finditer(run.log)}
        last = {k: max(c for a, c in stored.items() if a // MIB == k + 1) for k in range(KERNELS)}
        late = [(k, completed.get(k), last[k]) for k in range(KERNELS) if completed.get(k, -1) < last[k]]
        dram = [(m[1], m[2], (int(m[4]), m[3])) for m in DRAM.finditer(run.log)]
        acked = {key: i for i, (tag, _, key) in enumerate(dram) if tag == "DRAM_WRITE_ACK"}
        written = {key for tag, opcode, key in dram if tag == "REQ_TO_DRAM" and opcode == "MOVOUT"}
        unacked = [key for i, (tag, opcode, key) in enumerate(dram) if tag == "REQ_TO_DRAM" and opcode == "MOVIN"
                   and key in written and i < acked.get(key, len(dram))]
        if len(stored) != KERNELS * ITEMS * REQUESTS or early or late or len(acked) != len(stored) or unacked:
            run.fail(f"expected {KERNELS * ITEMS * REQUESTS} stored addresses, each loaded after its store, each "
                     f"write acked before a load of its address enters the DRAM model, and each kernel completed "
                     f"after its last store; got {len(stored)}, {len(early)} loads first (address, load, store; the "
                     f"first {early[:3]}), {len(acked)} writes acked, {len(unacked)} loads into the DRAM model before "
                     f"the ack (channel, address; the first {unacked[:3]}), (kernel, completed, last store) {late}")
    return case


def without_a_store(t):
    """Two kernels of the chain that store nothing, so that a compute is the last thing each does, with a DEVICE_SYNC
    after each: each kernel's completion line is logged, before the DEVICE_SYNC after it returns."""
    run = launch(t, 2, sync_between=True, defines=("STORE=0",))
    order = [m[0] for m in re.finditer(rf"Kernel \d+ has completed|{SYNCED}", run.log)]
    if order != ["Kernel 0 has completed", SYNCED, "Kernel 1 has completed", SYNCED]:
        run.fail(f"expected kernel 0's completion, a sync, kernel 1's completion, a sync; got {order}")


CASES = [*(Case(f"chain.{cores}_cores", chain(cores)) for cores in (1, 2)), Case("without_a_store", without_a_store)]
