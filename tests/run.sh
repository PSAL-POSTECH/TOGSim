#!/bin/bash
# Run every test under tests/ against a built tree and name the ones that failed; exit 0 iff none did.
#     tests/run.sh [build dir]    (default: this tree's build/, where bin/Simulator is)
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$(cd "${1:-$HERE/../build}" && pwd)" || exit 2
failed=()
run() {
  echo "=== $1"
  "${@:2}" || failed+=("$1")
}
run tensor_tiling_check "$HERE/tensor_tiling_check.sh" "$BUILD"
run semaphore_batch "$HERE/semaphore_batch/run_k1.sh" "$BUILD/bin/Simulator"
run sram_pieces "$HERE/sram_pieces/run_pieces.sh" "$BUILD/bin/Simulator"
run core_serial "$HERE/core_serial/run_core_serial.sh" "$BUILD/bin/Simulator"
if [ ${#failed[@]} -ne 0 ]; then echo "FAILED: ${failed[*]}"; exit 1; fi
echo "ALL PASS"
