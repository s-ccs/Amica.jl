#!/usr/bin/env bash
# Amica.jl GPU benchmark installer - run this on the CUDA machine.
#
# Sets up:
#   1. Julia: the `benchmark` project (local Amica + CUDA + JSON3)
#   2. Python: venv at benchmark/.venv with jamica + JAX (CUDA build)
set -euo pipefail
cd "$(dirname "$0")"

echo "== Julia: benchmark project (local Amica + CUDA + JSON3) =="
julia --project=. -e 'using Pkg; Pkg.develop(path=abspath("..")); Pkg.instantiate()'

echo "== Python: venv with jamica + JAX CUDA =="
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install jamica "jax[cuda12]"
.venv/bin/python -c 'import jax; print("JAX devices:", jax.devices())'

echo
echo "Done. Run the benchmark with:"
echo "  julia --project=benchmark benchmark/benchmark.jl"
echo "  julia --project=benchmark benchmark/compare.jl"
