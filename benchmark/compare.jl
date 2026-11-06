#!/usr/bin/env julia

# Joins the orchestrator's timing/memory results (benchmarks.json) with the
# per-implementation LL results (ll/*.json) into one table, and checks that
# all implementations agree on final log-likelihood for each (dataset,
# device, precision) group. If they don't, the timing comparison is not
# meaningful.

using JSON3
using Statistics

const BENCH_DIR = @__DIR__
# implementations now agree to ~1e-4 on the same data (f64); a larger drift
# indicates a schedule/preprocessing/data-loading mismatch
const LL_WARN_TOL = 0.25

struct Row
    dataset::String
    device::String
    precision::String
    implementation::String
    threads::Int
    n_iter::Int
    n_iter_run::Int
    fit_time_s::Float64
    ll_final::Union{Nothing,Float64}
    max_rss_mib::Int
    max_gpu_mib::Union{Nothing,Int}
    runs::Int
end

function load_bench(path)
    isfile(path) || return nothing
    return JSON3.read(read(path, String))
end

function load_ll(implementation::String, dataset::String, device::String, precision::String)
    path = joinpath(BENCH_DIR, "ll", "$(implementation)_$(dataset)_$(device)_$(precision).json")
    isfile(path) || return nothing
    return JSON3.read(read(path, String))
end

function gather_rows()
    bench = load_bench(joinpath(BENCH_DIR, "benchmarks.json"))
    bench === nothing && error("No benchmarks.json found - run benchmark/benchmark.jl first")

    rows = Row[]
    for (key, samples) in pairs(bench)
        for s in samples
            cfg = s.params
            impl = String(cfg.implementation)
            ll = load_ll(impl, String(cfg.dataset), String(cfg.device), String(cfg.precision))
            ll_final = ll === nothing ? nothing : (ll.ll_final === nothing ? nothing : Float64(ll.ll_final))
            n_iter_run = ll === nothing ? -1 : Int(ll.n_iter_run)
            fit_time_s = ll === nothing ? NaN : Float64(ll.fit_time_s)
            push!(
                rows,
                Row(
                    String(cfg.dataset),
                    String(cfg.device),
                    String(cfg.precision),
                    impl,
                    Int(cfg.threads),
                    Int(cfg.n_iter),
                    n_iter_run,
                    fit_time_s,
                    ll_final,
                    Int(s.max_cpu_rss_mib),
                    s.max_gpu_memory_mib === nothing ? nothing : Int(s.max_gpu_memory_mib),
                    length(samples),
                ),
            )
        end
    end
    return rows
end

function fmt_ll(ll)
    ll === nothing && return "n/a"
    return string(round(ll; digits = 4))
end

function main()
    rows = gather_rows()
    isempty(rows) && error("benchmarks.json contains no samples")

    groups = Dict{Tuple{String,String,String},Vector{Row}}()
    for r in rows
        push!(get!(groups, (r.dataset, r.device, r.precision), Row[]), r)
    end

    problems = String[]

    for ((dataset, device, precision), grp) in sort(collect(pairs(groups)); by = first)
        println("\n=== $dataset / $device / $precision ===")
        println("impl    threads  ms/iter    fit_s   RSS MiB  GPU MiB   LL final")
        for r in sort(grp; by = x -> (x.implementation, x.threads))
            ms_per_iter = isnan(r.fit_time_s) ? "n/a" : string(round(r.fit_time_s / r.n_iter * 1000; digits = 2))
            fit_s = isnan(r.fit_time_s) ? "n/a" : string(round(r.fit_time_s; digits = 2))
            gpu = r.max_gpu_mib === nothing ? "n/a" : string(r.max_gpu_mib)
            println(
                rpad(r.implementation, 8) *
                rpad(r.threads, 7) *
                rpad(ms_per_iter, 9) *
                rpad(fit_s, 9) *
                rpad(r.max_rss_mib, 9) *
                rpad(gpu, 10) *
                fmt_ll(r.ll_final) *
                "  [$(r.runs) runs]",
            )
        end

        # LL equivalence gate
        lls = [r.ll_final for r in grp if r.ll_final !== nothing]
        impls_with_ll = sort(unique([r.implementation for r in grp if r.ll_final !== nothing]))
        if length(lls) >= 2 && length(impls_with_ll) >= 2
            spread = maximum(lls) - minimum(lls)
            if spread > LL_WARN_TOL
                push!(
                    problems,
                    "$dataset/$device/$precision: final LL spread $(round(spread; digits = 4)) across " *
                    "$(join(impls_with_ll, ", ")) exceeds tolerance $(LL_WARN_TOL) - timing comparison is not meaningful",
                )
            else
                println("LL agreement: spread $(round(spread; digits = 4)) across $(join(impls_with_ll, ", ")) (ok)")
            end
        else
            println("LL agreement: only one implementation has LL results ($(join(impls_with_ll, ", ")))")
        end

        # early exit check: both sides must have run all iterations
        for r in grp
            if r.n_iter_run != -1 && r.n_iter_run != r.n_iter
                push!(
                    problems,
                    "$dataset/$device/$precision [$(r.implementation)]: ran $(r.n_iter_run)/$(r.n_iter) iterations " *
                    "(early exit) - timing comparison is not meaningful",
                )
            end
        end
    end

    println("\n========== Comparison Report ==========")
    if isempty(problems)
        println("All LL checks passed.")
    else
        println("Problems:")
        for p in problems
            println(" - $p")
        end
    end
    println("=======================================")
    return isempty(problems) ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
