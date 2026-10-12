"""A spad buffer is as large as the TOG states it, however many DMAs fill or drain it; an unsized one is refused."""
from togtest import Case

FOOTPRINT = {
    "piecewise_fill": dict(spad_footprint=32768, max_dispatch=1),
    "indirect_store": dict(spad_footprint=16640, max_dispatch=2),
    "two_stores_one_buffer": dict(spad_footprint=32768, max_dispatch=1),
    "state_stored_each_trip": dict(spad_footprint=16384, max_dispatch=2),
    "computed_state_stored_each_trip": dict(spad_footprint=32768, max_dispatch=1),
}
REFUSED = {
    "refused.buffer_beyond_size_table": dict(defines=(), saying="spad buffer 1 is DMA'd but the TOG states no size"),
    "refused.no_size_table": dict(defines=("NO_TABLE",), saying="states no spad buffer sizes"),
}


def replay(t, source, defines=()):
    """The TOG on tpuv6e with a 48 KB spad and no array split (it brings no unit table), every tile 200 cycles."""
    return t.simulate(t.tog(source, defines), t.machine(core_spad_size_kb=48, systolic_array_split="none"),
                      cycles=[200])


def footprint(name, spad_footprint, max_dispatch):
    """Both work-items of <name>_tog.cpp are scheduled with that footprint in bytes, that many of them at once."""
    def case(t):
        run = replay(t, f"{name}_tog.cpp").expect_ok()
        if run.scheduled() != [(spad_footprint, max_dispatch)] * 2:
            run.fail(f"expected two work-items scheduled with (spad_footprint, max_dispatch) = "
                     f"({spad_footprint}, {max_dispatch}), got {run.scheduled()}")
    return case


def refused(defines, saying):
    return lambda t: replay(t, "unsized_buffer_tog.cpp", defines).expect_refused(saying, exit=1)


CASES = [Case(name, footprint(name, **expected)) for name, expected in FOOTPRINT.items()] + [
    Case(name, refused(**how)) for name, how in REFUSED.items()]
