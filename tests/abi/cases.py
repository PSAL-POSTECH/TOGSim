"""A TOG built against another TOG ABI is refused by name and version; the same TOG at this ABI replays."""
from togtest import Case

ABI = 16
REFUSED = {
    "refused.previous_version": dict(
        define="PREVIOUS_VERSION", saying=f"it was built against TOG ABI {ABI - 1}, TOGSim expects ABI {ABI}"),
    "refused.symbol_of_abi_15": dict(
        define="OLD_SYMBOL", saying="it was built against ABI 15 (it embeds togsim_producer_abi_version, the symbol "
                                    f"of ABI 15 and earlier), TOGSim expects ABI {ABI}; rebuild the TOG"),
    "refused.no_version_symbol": dict(
        define="NO_SYMBOL", saying=f"it embeds no togsim_abi_version (built without togsim_runtime.h); TOGSim "
                                   f"expects ABI {ABI}"),
}


def replay(t, *defines):
    """versioned_tog.cpp on tpuv6e without the array split, its one compute tile 100 cycles."""
    return t.simulate(t.tog("versioned_tog.cpp", defines, tensors=()), t.machine(systolic_array_split="none"),
                      cycles=[100])


def this_version_replays(t):
    """The control: with the header's version the TOG runs its compute, so the refusals below are the version's."""
    run = replay(t).expect_ok()
    if [c.cycles for c in run.computes()] != [100]:
        run.fail(f"expected the TOG's one compute of 100 cycles to finish, got {run.computes()}")


def refused(define, saying):
    """The loader refuses the TOG on stderr with `saying`, and TOGSim exits 1."""
    return lambda t: replay(t, define).expect_refused(f"togsim: refusing TOG {t.work_dir}/tog.so: {saying}", exit=1)


CASES = [Case("this_version_replays", this_version_replays)] + [
    Case(name, refused(**how)) for name, how in REFUSED.items()]
