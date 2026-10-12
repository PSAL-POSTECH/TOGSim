"""A barrier waits every async load signaled on its semaphore; a TOG that breaks the semaphore contract is refused."""
from togtest import Case, write_table

TOG = "barrier_waits_both_loads_tog.cpp"
REFUSED = {
    "refused.wait_with_wrong_count": dict(define="WRONG_COUNT", saying="wait expects another element count"),
    "refused.wait_on_unsignaled_semaphore": dict(define="UNSIGNALED_SEM", saying="wait on a semaphore never signaled"),
    "refused.load_batch_never_waited": dict(define="NO_WAIT", saying="unwaited load batch"),
}


def replay(t, *defines):
    """The TOG on tpuv6e without the array split (it brings no unit table), its one compute tile 200 cycles."""
    return t.simulate(t.tog(TOG, defines), t.machine(systolic_array_split="none"), cycles=[200])


def barrier_waits_both_loads(t):
    """The second load still arrives first (else the scenario is lost), and the barrier ends after both."""
    run = replay(t).expect_ok()
    big, small = run.when("DRAM_RESP_DONE", 0), run.when("DRAM_RESP_DONE", 1)
    barrier, compute = run.when("INST_FINISHED", 2), run.when("INST_FINISHED", 3)
    seen = f"strided load arrives @{big}, small load @{small}, barrier finishes @{barrier}, compute @{compute}"
    if not small < big:
        run.fail(f"expected the small load to arrive before the strided one (the scenario): {seen}")
    if barrier < max(big, small):
        run.fail(f"expected the barrier to finish after both loads arrived: {seen}")


def session_wrong_count_refused_at_launch(t):
    """--models_list: with a spad size the TOG is built once while it is loaded, so the contract it breaks is the
    error of the LAUNCH_KERNEL command; without one (core_spad_size_kb: 0) it is that of the DEVICE_SYNC that
    runs the kernel."""
    home = t.tog(TOG, ("WRONG_COUNT",), into="k").parent
    write_table(home / "tog_cycles.tsv", [(200, 0)])
    stream = t.work_dir / "commands.txt"
    stream.write_text(f"LAUNCH_KERNEL,0,0,0,{home}/kernel,{home}/attributes,0\nDEVICE_SYNC,0,0,0,-,-,0\n")
    for spad, command in (({}, "LAUNCH_KERNEL"), (dict(core_spad_size_kb=0), "DEVICE_SYNC")):
        machine = t.machine(systolic_array_split="none", **spad)
        run = t.execute("Simulator", ["--config", machine, "--models_list", stream, "--log_level", "trace"])
        run.expect_refused(f"Error processing command 0 (type: {command}): [TOGSim] wait expects another element "
                           "count", exit=1)


def refused(define, saying):
    return lambda t: replay(t, define).expect_refused(saying, exit=1)


CASES = [Case("barrier_waits_both_loads", barrier_waits_both_loads)] + [
    Case(name, refused(**how)) for name, how in REFUSED.items()] + [
    Case("session.refused.wait_with_wrong_count", session_wrong_count_refused_at_launch)]
