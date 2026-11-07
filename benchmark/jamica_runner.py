#!/usr/bin/env python3
"""jamica runner for the Amica.jl benchmark harness.

Takes the same 7-field config as julia_runner.jl:

    dataset implementation device block_size precision threads n_iter

Does a 1-iteration JIT warm-up fit (excluded from timing), then the timed
max_iter fit, and writes benchmark/ll/jamica_<dataset>_<device>_<precision>.json
with fit_time_s, ll_final and the full log-likelihood history (same schema as
snesmaeili/jamica-benchmark's runners).
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
USAGE = "config format (7 fields): dataset implementation device block_size precision threads n_iter"
DATASETS = {"memorize": ("Memorize.fdt", 71), "small": ("small.fdt", 19)}
SEED = 42


def parse_config(argv):
    tokens = " ".join(argv).split()
    if len(tokens) != 7:
        sys.exit(USAGE)
    dataset, implementation, device, block_size, precision, threads, n_iter = [t.lower() for t in tokens]
    if dataset not in DATASETS:
        sys.exit(f"Invalid dataset '{dataset}' (expected memorize or small)")
    if implementation != "jamica":
        sys.exit(f"jamica_runner expects implementation=jamica, got '{implementation}'")
    if device not in ("cpu", "gpu"):
        sys.exit(f"Invalid device '{device}' (expected cpu or gpu)")
    if precision not in ("float32", "float64"):
        sys.exit(f"Invalid precision '{precision}' (expected float32 or float64)")
    if int(block_size) < 1 or int(threads) < 1 or int(n_iter) < 1:
        sys.exit("block_size, threads and n_iter must be positive integers")
    return {
        "dataset": dataset,
        "device": device,
        "block_size": int(block_size),
        "precision": precision,
        "threads": int(threads),
        "n_iter": int(n_iter),
    }


def make_config(cfg, n_iter):
    from jamica import AmicaConfig

    # matched to Amica.jl's defaults (src/learning_rate.jl); Newton stays off on
    # both sides because newt_start=50 > n_iter=40
    return AmicaConfig(
        num_models=1,
        num_mix_comps=3,
        dtype=cfg["precision"],
        max_iter=n_iter,
        lrate=0.1,
        lratefact=0.5,
        rholrate=0.05,
        rholratefact=0.1,
        minlrate=1e-12,
        max_decs=3,
        max_incs=5,
        do_newton=True,
        newt_start=50,
        newt_ramp=10,
        newtrate=0.5,
        do_mean=True,
        do_sphere=True,
        sphere_type="zca",
        pcakeep=None,
        mineig=1e-15,
        chunk_size=cfg["block_size"],
    )


def read_fdt(path, ncols, dtype):
    # .fdt stores one time point per ncols consecutive floats (Fortran amica reads
    # direct records of num_channels floats, same as read_fdt(transpose=true));
    # reshape (samples, channels), then transpose to jamica's (channels, samples)
    raw = np.fromfile(path, dtype=np.float32)
    return raw.reshape(-1, ncols).T.astype(dtype)


def main():
    cfg = parse_config(sys.argv[1:])
    # force CPU before jax is imported, so cpu runs stay on CPU on CUDA machines
    if cfg["device"] == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
        # pin to 1 core: XLA spins a thread pool across all cores on jax 0.12
        # (--xla_cpu_multi_thread_eigen is ignored there), which would make the
        # CPU comparison unfair vs julia threads=1
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, {0})

    import jax
    from jamica import Amica

    name, ncols = DATASETS[cfg["dataset"]]
    data_path = BENCH_DIR.parent / "test" / "input" / name
    X = read_fdt(data_path, ncols, cfg["precision"])

    # JIT warm-up: same data shape, 1 iteration (compiles all kernels)
    Amica(make_config(cfg, 1), random_state=0).fit(X)

    model = Amica(make_config(cfg, cfg["n_iter"]), random_state=SEED)
    t0 = time.perf_counter()
    result = model.fit(X)
    fit_time = time.perf_counter() - t0

    ll = [float(v) for v in np.asarray(result.log_likelihood).ravel()]
    out = {
        "implementation": "jamica",
        "dataset": cfg["dataset"],
        "device": cfg["device"],
        "precision": cfg["precision"],
        "block_size": cfg["block_size"],
        "threads": cfg["threads"],
        "n_iter": cfg["n_iter"],
        "n_iter_run": int(result.n_iter),
        "fit_time_s": fit_time,
        "ll_final": ll[-1] if ll else None,
        "ll_history": ll,
        "seed": SEED,
        "jax_backend": jax.default_backend(),
    }

    ll_dir = BENCH_DIR / "ll"
    ll_dir.mkdir(exist_ok=True)
    path = ll_dir / f"jamica_{cfg['dataset']}_{cfg['device']}_{cfg['precision']}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")

    ll_str = "nothing" if out["ll_final"] is None else f"{out['ll_final']:.4f}"
    print(f"jamica [{jax.default_backend()}] fit_time_s={fit_time:.3f} n_iter_run={out['n_iter_run']} ll_final={ll_str} -> {path}")


if __name__ == "__main__":
    main()
