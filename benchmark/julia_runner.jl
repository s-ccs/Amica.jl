using Amica
using JSON3
using LinearAlgebra
using Random

# the runner's threads config is julia Threads; BLAS would otherwise spin its
# own pool on the small gemms (measured ~4x user time vs wall) - keep 1 thread
BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "..", "test", "util.jl"))

const RUNNER_PATH = abspath(@__FILE__)
const BENCH_DIR = @__DIR__
const USAGE = "config format (7 fields): dataset implementation device block_size precision threads n_iter"
const SEED = 42

function parse_config(args::Vector{String})
    tokens = isempty(args) ? String[] : split(strip(join(args, " ")))
    length(tokens) == 7 || error(USAGE)

    dataset = lowercase(tokens[1])
    dataset in ("memorize", "small") || error("Invalid dataset '$dataset' (expected memorize or small)")
    implementation = lowercase(tokens[2])
    implementation == "julia" || error("julia_runner expects implementation=julia, got '$implementation'")
    device = lowercase(tokens[3])
    device in ("cpu", "gpu") || error("Invalid device '$device' (expected cpu or gpu)")
    block_size = parse_positive_int("block_size", tokens[4])
    precision_str = lowercase(tokens[5])
    precision_str in ("float32", "float64") || error("Invalid precision '$precision_str' (expected float32 or float64)")
    precision = precision_str == "float32" ? Float32 : Float64
    num_threads = parse_positive_int("threads", tokens[6])
    n_iter = parse_positive_int("n_iter", tokens[7])

    config = join((dataset, "julia", device, string(block_size), precision_str, string(num_threads), string(n_iter)), " ")

    return (dataset, device, block_size, precision, precision_str, num_threads, n_iter, config)
end

function parse_positive_int(name::String, value::AbstractString)
    iv = tryparse(Int, value)
    (iv === nothing || iv <= 0) && error("Invalid value for $name: '$value' (expected positive integer)")
    return iv
end

function dataset_spec(dataset::String)
    if dataset == "memorize"
        return (path = joinpath(BENCH_DIR, "..", "test", "input", "Memorize.fdt"), ncols = 71)
    else
        return (path = joinpath(BENCH_DIR, "..", "test", "input", "small.fdt"), ncols = 19)
    end
end

function resolve_array_type(device::AbstractString)
    device == "cpu" && return Array
    isdefined(Main, :CUDA) || error("CUDA device requested, but CUDA.jl is not loaded")
    return getfield(Main, :CUDA).CuArray
end

function ensure_julia_threads!(num_threads::Int, config::String)
    Threads.nthreads() == num_threads && return false

    if get(ENV, "AMICA_JULIA_RUNNER_REEXEC", "0") == "1"
        error(
            "Requested $num_threads Julia threads, but process still has $(Threads.nthreads()) " *
            "after re-exec attempt.",
        )
    end

    julia_bin = joinpath(Sys.BINDIR, Base.julia_exename())
    cmd = `$julia_bin --threads=$num_threads --project=$BENCH_DIR $RUNNER_PATH $config`

    println("Re-launching julia_runner with threads=$num_threads (current=$(Threads.nthreads()))")
    run(setenv(cmd, "AMICA_JULIA_RUNNER_REEXEC" => "1"))
    return true
end

function run_benchmark(dataset, device, block_size, precision, precision_str, num_threads, n_iter)
    Threads.nthreads() == num_threads || error(
        "Julia process has $(Threads.nthreads()) threads, but config requested $num_threads. ",
    )

    array_type = resolve_array_type(device)
    spec = dataset_spec(dataset)
    isfile(spec.path) || error("Dataset file not found: $(spec.path)")
    data = read_fdt(spec.path; ncols = spec.ncols, T = Float32, transpose = true, OutType = precision)
    N, n = size(data)

    println(
        "Configuration: dataset=$(dataset), device=$(device), precision=$(precision_str), " *
        "block_size=$(block_size), threads=$(num_threads), n_iter=$(n_iter)",
    )

    # warm-up run: compiles all kernels (incl. GPU) so the timed run measures fitting only
    Random.seed!(SEED)
    warmup = SingleModelAmica(
        precision,
        ncomps = n,
        nsamples = N,
        m = 3,
        ArrayType = array_type,
        block_size = block_size,
        num_threads = num_threads,
    )
    Amica.amica!(warmup, data; maxiter = 1, show_progress = false)

    Random.seed!(SEED)
    myAmica = SingleModelAmica(
        precision,
        ncomps = n,
        nsamples = N,
        m = 3,
        ArrayType = array_type,
        block_size = block_size,
        num_threads = num_threads,
    )

    fit_time = @elapsed Amica.amica!(myAmica, data; maxiter = n_iter, show_progress = true)

    ll_history = Float64[x for x in myAmica.LL if !isnan(x)]
    result = Dict(
        "implementation" => "julia",
        "dataset" => dataset,
        "device" => device,
        "precision" => precision_str,
        "block_size" => block_size,
        "threads" => num_threads,
        "n_iter" => n_iter,
        "n_iter_run" => length(ll_history),
        "fit_time_s" => fit_time,
        "ll_final" => isempty(ll_history) ? nothing : ll_history[end],
        "ll_history" => ll_history,
        "seed" => SEED,
    )

    ll_dir = joinpath(BENCH_DIR, "ll")
    mkpath(ll_dir)
    # threads in the key: julia t1 and t16 must not overwrite each other's timing
    open(joinpath(ll_dir, "julia_$(dataset)_$(device)_$(precision_str)_t$(num_threads).json"), "w") do io
        JSON3.write(io, result)
        write(io, '\n')
    end

    println(
        "julia fit_time_s=$(round(fit_time; digits = 3)) n_iter_run=$(length(ll_history)) " *
        "ll_final=$(isempty(ll_history) ? "nothing" : round(ll_history[end]; digits = 4))",
    )
    return nothing
end

function main(args::Vector{String})
    (dataset, device, block_size, precision, precision_str, num_threads, n_iter, config) = parse_config(args)

    if ensure_julia_threads!(num_threads, config)
        return nothing
    end

    if device == "gpu"
        try
            @eval import CUDA
        catch err
            error("Failed to load CUDA: $(sprint(showerror, err))")
        end
        return Base.invokelatest(
            run_benchmark,
            dataset,
            device,
            block_size,
            precision,
            precision_str,
            num_threads,
            n_iter,
        )
    end

    return run_benchmark(dataset, device, block_size, precision, precision_str, num_threads, n_iter)
end

main(ARGS)
