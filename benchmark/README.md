# GPU benchmark: Amica.jl vs jamica

Timing/memory benchmark harness for Amica.jl against the JAX-based Python
implementation [jamica](https://github.com/snesmaeili/jamica). Ported from
[`s-ccs/2025_MSc_AMICA/scripts`](https://github.com/s-ccs/2025_MSc_AMICA/tree/main/scripts)
(same-author harness matching Amica.jl's API and datasets); the jamica runner
records the same result schema as
[`snesmaeili/jamica-benchmark`](https://github.com/snesmaeili/jamica-benchmark).

## Setup (CUDA machine)

```bash
./benchmark/install_cuda_machine.sh
```

Creates the `benchmark` Julia project (local Amica + CUDA + JSON3) and a Python
venv at `benchmark/.venv` with jamica + JAX-CUDA. Requires an NVIDIA GPU with
CUDA 12.

## Run

```bash
# full suite (small + memorize; CPU f64 1/16 threads; GPU f64/f32)
julia --project=benchmark benchmark/benchmark.jl

# single config
julia --project=benchmark benchmark/benchmark.jl "small julia gpu 50000 float32 1 40"

# combined table + log-likelihood agreement check
julia --project=benchmark benchmark/compare.jl
```

## Config format

`dataset implementation device block_size precision threads n_iter`

- datasets: `small` (172704×19), `memorize` (319500×71) from `test/input/`
- implementations: `julia`, `jamica`
- precision: `float32` / `float64`

## How it works

- Each sample runs in a fresh subprocess; `benchmark.jl` samples wall time, CPU
  RSS (`/proc`) and GPU memory (NVML, process tree) while it runs, and records
  everything in `benchmark/benchmarks.json` (suite mode resumes/prunes).
- Both runners do a **1-iteration warm-up fit** (JIT/compile excluded) and
  report `fit_time_s` for the timed run only; ms/iter = `fit_time_s / n_iter`.
- Each runner writes `benchmark/ll/<impl>_<dataset>_<device>_<precision>.json`
  with the final log-likelihood and full LL history.
- `compare.jl` joins timings and LL results and warns when final LLs drift
  apart between implementations — in that case the timing comparison is not
  meaningful.

## Fairness notes

- Fixed maxiter (40) on both sides: identical work per sample; the LL history
  confirms both ran all iterations.
- Same data: `.fdt` stores one time point per `n_channels` consecutive floats
  (the Fortran amica convention) and both runners read it that way.
- Same fixed schedule on both sides (lrate 0.1, lratefact 0.5, same Newton
  settings — Newton stays off because `newt_start=50 > 40` iterations), same
  data, same block size.
- Sphering matched: both do ZCA whitening on mean-removed data, full rank.
- Init is seeded per side (`Random.seed!(42)` / `random_state=42`); RNG streams
  differ across languages, so final LLs agree only approximately.

## Results

`benchmark/benchmarks.json` and `benchmark/ll/*.json` (gitignored).
