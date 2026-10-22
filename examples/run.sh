#!/usr/bin/env bash
# Build and run the example TOGs against a built tree: examples/run.sh [hello|async_barrier|tiles|gemm|session|mistakes]...
# Each command is echoed after "+"; under it come the lines of its log that carry the result, without their
# timestamp, and "exit N" when the Simulator did not exit 0. Every <name>/expected.txt is this output for <name>.
set -u
cd -P "$(dirname "$0")/.."
SIM=build/bin/Simulator
V6E=configs/tpuv6e.yml
EXAMPLES="hello async_barrier tiles gemm session mistakes"
ulimit -c 0

# tog <g++ arguments>: compile a TOG against include/togsim.hpp (TOGSim API v1.0.0). It links against nothing.
tog() {
  echo "+ g++ -std=c++20 -shared -fPIC -O1 -Iinclude $*"
  ${CXX:-g++} -std=c++20 -shared -fPIC -O1 -Iinclude "$@"
}

# sim <filter> <Simulator arguments>: simulate, and print what <filter> (one of the functions below) keeps of the log.
sim() {
  local filter=$1 status
  shift
  echo "+ $SIM $*"
  "$SIM" "$@" 2>&1 | sed -E -e 's/^\[[0-9-]+ [0-9:.]+\] //' -e "s#$PWD/##g" | "$filter"
  status=${PIPESTATUS[0]}
  [ "$status" -eq 0 ] || echo "exit $status"
}

# machine <out.yml> <base.yml> <"key: value" | "no key">...: a copy of a shipped config with those keys set or
# left out, and with its ../configs paths made absolute so that the copy works outside configs/.
machine() {
  local out=$1 base=$2 key_value what
  shift 2
  what=$(printf '%s, ' "$@")
  echo "# $out: $base with ${what%, }"
  sed "s#: \.\./configs/#: $PWD/configs/#" "$base" > "$out"
  for key_value in "$@"; do
    case $key_value in
      "no "*) sed -i "/^${key_value#no }:/d" "$out" ;;
      *) sed -i "/^${key_value%%:*}:/d" "$out"; echo "$key_value" >> "$out" ;;
    esac
  done
}

# Filters of a --log_level trace run. timeline: every instruction, and when an async load has sent its requests
# and has its data. instruction_0: every event of instruction 0. computes: the computes as issued. scheduling:
# computes, and the work-items as handed to a core. unrefused: what a run of a broken TOG that is not refused shows,
# the line that names its held buffers among it.
timeline() { grep -E 'TILE_SCHEDULED|INST_(ISSUED|FINISHED)|ASYNC_DMA_ISSUE|DRAM_RESP_DONE|Total execution cycles'; }
instruction_0() { grep -F 'INST_ID=0]'; }
computes() { grep -E 'INST_ISSUED.*COMP|Total execution cycles'; }
scheduling() { grep -E 'TILE_SCHEDULED|INST_ISSUED.*COMP|Total execution cycles'; }
unrefused() { awk '/\[warning\]|bytes for the kernel|INST_ISSUED/ || (final && /responses\)|Total execution cycles/); /Simulation finished/ { final = 1 }'; }

# Filters of any run. statistics: the DRAM summary and the final statistics. result: warnings, the computes as
# issued, the unit lines of the final statistics and the total. kernels: the kernels and computes of a session.
# first_error: the line that says why a run was refused. whole: the log as it is.
statistics() { awk '/\[DRAM\] (Per-channel average|channel 0 \||channels 0\.\.)/ || final; /Simulation finished/ { end = 1 } end && /Instructions count/ { final = 1; print }'; }
result() { awk '/\[warning\]|INST_ISSUED.*COMP/ || (final && /utilization\(%\)|Total execution cycles/); /Simulation finished/ { final = 1 }'; }
kernels() { grep -E 'Enqueued kernel_id|INST_ISSUED.*COMP|Total execution cycles'; }
first_error() { grep -m 1 -E '^togsim(\.hpp)?: |\[error\]|what\(\):|does not exist!'; }
whole() { cat; }

# One work-item: its timeline, one instruction's events and the final statistics on tpuv6e, then a whole log on a
# config that needs no other file.
hello() {
  local dir=examples/hello
  local run="--config $V6E --tog_so $dir/tog.so --cycle_table $dir/tog_cycles.tsv --unit_table $dir/tog_units.tsv"
  tog $dir/hello_tog.cpp -o $dir/tog.so
  sim timeline $run --log_level trace
  sim instruction_0 $run --log_level trace
  sim statistics $run
  sim whole --config examples/minimal.yml --tog_so $dir/tog.so --cycle_table $dir/tog_cycles.tsv
}

# Two async loads and their barrier, then the same loads made synchronous; then both forms again with a first
# compute that needs only one of the two loads.
async_barrier() {
  local dir=examples/async_barrier
  local tables="--cycle_table $dir/tog_cycles.tsv --unit_table $dir/tog_units.tsv"
  tog $dir/async_barrier_tog.cpp -o $dir/tog.so
  sim timeline --config $V6E --tog_so $dir/tog.so $tables --log_level trace
  tog -DSYNC $dir/async_barrier_tog.cpp -o $dir/sync.so
  sim timeline --config $V6E --tog_so $dir/sync.so $tables --log_level trace
  tog -DTWO_STEPS $dir/async_barrier_tog.cpp -o $dir/two_steps.so
  sim computes --config $V6E --tog_so $dir/two_steps.so $tables --log_level trace
  tog -DTWO_STEPS -DSYNC $dir/async_barrier_tog.cpp -o $dir/two_steps_sync.so
  sim computes --config $V6E --tog_so $dir/two_steps_sync.so $tables --log_level trace
}

# Eight work-items (tog_shape.txt: 2 4) on one core, on two and on four; then on one core in a 48 KB spad, and
# with half the DRAM channels.
tiles() {
  local dir=examples/tiles config
  tog $dir/tiles_tog.cpp -o $dir/tog.so
  machine $dir/two_cores.gen.yml $V6E "num_cores: 2"
  machine $dir/four_cores.gen.yml $V6E "num_cores: 4"
  machine $dir/small_spad.gen.yml $V6E "no vpu_spad_size_kb_per_lane" "core_spad_size_kb: 48"
  machine $dir/half_dram.gen.yml $V6E "dram_channels: 16"
  for config in $V6E $dir/two_cores.gen.yml $dir/four_cores.gen.yml $dir/small_spad.gen.yml $dir/half_dram.gen.yml; do
    sim scheduling --config $config --tog_so $dir/tog.so --cycle_table $dir/tog_cycles.tsv --unit_table $dir/tog_units.tsv --log_level trace
  done
}

# What the unit table does to a compute: 512 rows over two arrays, 256 rows, one array, 256 rows on 128 lanes, and
# no array split.
gemm() {
  local dir=examples/gemm
  local run="--tog_so $dir/tog.so --cycle_table $dir/tog_cycles.tsv --log_level trace"
  tog $dir/gemm_tog.cpp -o $dir/tog.so
  sim result --config $V6E $run --unit_table $dir/tog_units.tsv
  sim result --config $V6E $run --unit_table $dir/tog_units_128_rows.tsv
  machine $dir/one_array.gen.yml $V6E "num_systolic_array_per_core: 1"
  sim result --config $dir/one_array.gen.yml $run --unit_table $dir/tog_units.tsv
  machine $dir/64_lanes.gen.yml $V6E "vpu_num_lanes: 64"
  sim result --config $dir/64_lanes.gen.yml $run --unit_table $dir/tog_units_128_rows.tsv
  machine $dir/no_split.gen.yml $V6E "systolic_array_split: none" "no systolic_array_unit"
  sim result --config $dir/no_split.gen.yml $run
}

# Two kernels from one command stream: one after the other on one partition, then side by side on two.
session() {
  local dir=examples/session kernel
  for kernel in k0 k1; do
    tog examples/hello/hello_tog.cpp -o $dir/$kernel/tog.so
  done
  sim kernels --config $V6E --models_list $dir/commands.txt --log_level trace
  machine $dir/two_partitions.gen.yml $V6E "num_cores: 2" "num_partition: 2" "partition: {core_0: 0, core_1: 1}"
  sim kernels --config $dir/two_partitions.gen.yml --models_list $dir/commands_two_partitions.txt --log_level trace
}

# The valid TOG and each way to break it, refused by togsim.hpp or by TOGSim; the valid TOG with each wrong input or
# machine that is refused; then what is not refused: two mistakes that a warning names and one that reads as data
# already on the chip, which an info line names and a spad without room for it refuses.
mistakes() {
  local dir=examples/mistakes broken
  local tables="--cycle_table $dir/tog_cycles.tsv --unit_table $dir/tog_units.tsv"
  tog $dir/refused_tog.cpp -o $dir/valid.so
  sim result --config $V6E --tog_so $dir/valid.so $tables
  for broken in NOT_A_TOG UNSIZED_BUFFER NO_WAIT EMPTY_WAIT TILE_PAST_TABLE; do
    tog -D$broken $dir/refused_tog.cpp -o $dir/$broken.so
    sim first_error --config $V6E --tog_so $dir/$broken.so $tables
  done
  sim first_error --config $V6E --tog_so $dir/valid.so --unit_table $dir/tog_units.tsv
  sim first_error --config $V6E --tog_so $dir/valid.so --cycle_table $dir/tog_cycle.tsv --unit_table $dir/tog_units.tsv
  printf 'cycle\toverlapping\n600\t0\n' > $dir/cycles_header.gen.tsv
  sim first_error --config $V6E --tog_so $dir/valid.so --cycle_table $dir/cycles_header.gen.tsv --unit_table $dir/tog_units.tsv
  sim first_error --config $V6E --tog_so $dir/valid.so --cycle_table $dir/tog_cycles.tsv
  printf '0 TPUV6E_MXU rows 1 128\n' > $dir/spaces.gen.tsv
  sim first_error --config $V6E --tog_so $dir/valid.so --cycle_table $dir/tog_cycles.tsv --unit_table $dir/spaces.gen.tsv
  printf 'tile_id\tunit\tunit_of_work\tcapacity_per_cycle\tadmitted\n0\tTPUV6E_MXU\trows\t1\t128\n' > $dir/header.gen.tsv
  sim first_error --config $V6E --tog_so $dir/valid.so --cycle_table $dir/tog_cycles.tsv --unit_table $dir/header.gen.tsv
  printf '0\tVpu\tcycles\t1\t100\n' > $dir/no_array_unit.gen.tsv
  sim first_error --config $V6E --tog_so $dir/valid.so --cycle_table $dir/tog_cycles.tsv --unit_table $dir/no_array_unit.gen.tsv
  mkdir -p $dir/no_tensors && cp $dir/valid.so $dir/no_tensors/tog.so
  sim first_error --config $V6E --tog_so $dir/no_tensors/tog.so $tables
  echo "+ cd $dir"
  (cd $dir && SIM=../../$SIM sim first_error --config ../../$V6E --tog_so valid.so --cycle_table tog_cycles.tsv --unit_table tog_units.tsv)
  machine $dir/spad_16kb.gen.yml $V6E "no vpu_spad_size_kb_per_lane" "core_spad_size_kb: 16"
  sim first_error --config $dir/spad_16kb.gen.yml --tog_so $dir/valid.so $tables
  cp $V6E $dir/copied.gen.yml
  sim first_error --config $dir/copied.gen.yml --tog_so $dir/valid.so $tables
  grep -v '^num_cores:' examples/minimal.yml > $dir/no_num_cores.gen.yml
  sim first_error --config $dir/no_num_cores.gen.yml --tog_so $dir/valid.so --cycle_table $dir/tog_cycles.tsv
  tog $dir/silent_tog.cpp -o $dir/silent_valid.so
  sim unrefused --config $V6E --tog_so $dir/silent_valid.so $tables --log_level trace
  printf '1\tTPUV6E_MXU\trows\t1\t128\n1\tVpu\tcycles\t1\t100\n' > $dir/rows_of_tile_1.gen.tsv
  sim result --config $V6E --tog_so $dir/silent_valid.so --cycle_table $dir/tog_cycles.tsv --unit_table $dir/rows_of_tile_1.gen.tsv
  for broken in TENSOR_PAST_LIST CONSUMER_FIRST; do
    tog -D$broken $dir/silent_tog.cpp -o $dir/$broken.so
    sim unrefused --config $V6E --tog_so $dir/$broken.so $tables --log_level trace
  done
  sim first_error --config $dir/spad_16kb.gen.yml --tog_so $dir/CONSUMER_FIRST.so $tables
}

[ -x "$SIM" ] || { echo "$SIM is absent: build TOGSim first (README.md, Quick start)" >&2; exit 2; }
[ $# -gt 0 ] || set -- $EXAMPLES
for example; do
  case " $EXAMPLES " in
    *" $example "*) echo "== $example"; $example ;;
    *) echo "no example named $example" >&2; exit 2 ;;
  esac
done
