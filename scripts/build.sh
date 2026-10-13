#!/usr/bin/env bash
# Build go9 on Linux / macOS.
#   ./scripts/build.sh                         # uses pip-installed torch (pip install torch)
#   TORCH_ROOT=/path/to/libtorch ./scripts/build.sh
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -z "${TORCH_ROOT:-}" ]]; then
  TORCH_ROOT="$(python3 -c 'import torch, os; print(os.path.dirname(torch.__file__))')"
fi
echo "LibTorch: $TORCH_ROOT"
GEN=()
command -v ninja >/dev/null && GEN=(-G Ninja)
cmake -S . -B build "${GEN[@]}" -DCMAKE_BUILD_TYPE=Release -DTORCH_ROOT="$TORCH_ROOT"
cmake --build build --parallel
./build/go9_tests
./build/go9_crossval tests/data/openspiel_games.txt
./build/go9 selftest
echo
echo "Next:  ./build/go9 train --config configs/cpu.ini --run_dir runs/main"
echo "       ./build/go9 gui --run runs/main --open"
