#!/usr/bin/env bash
# Amica.jl GPU benchmark installer - run this on the CUDA machine.
#
# Sets up:
#   1. Julia: the `benchmark` project (local Amica + CUDA + JSON3)
#   2. Python: uv venv at benchmark/.venv with jamica + JAX (CUDA build)
set -euo pipefail
cd "$(dirname "$0")"

echo "== Julia: benchmark project (local Amica + CUDA + JSON3) =="
julia --project=. -e 'using Pkg; Pkg.develop(path=abspath("..")); Pkg.instantiate()'

echo "== Python: uv venv with jamica + JAX CUDA =="
uv venv .venv
uv pip install --python .venv/bin/python jamica "jax[cuda12]"
# fail loudly on CPU-only jax: jamica's gpu configs would silently run on CPU
.venv/bin/python -c 'import jax; d = jax.devices(); print("JAX devices:", d); assert jax.default_backend() != "cpu", "jax fell back to CPU - CUDA-enabled jaxlib missing; run: uv pip install --python .venv/bin/python \"jax[cuda12]\""'

echo
echo "Done. Run the benchmark with:"
echo "  julia --project=benchmark benchmark/benchmark.jl"
echo "  julia --project=benchmark benchmark/compare.jl"
