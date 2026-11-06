using JSON3
using Statistics

const libnvml = "libnvidia-ml.so.1"
const NVML_SUCCESS = 0
const NVML_ERROR_INSUFFICIENT_SIZE = 7
const NVML_VALUE_NOT_AVAILABLE = typemax(Culonglong)
const SAMPLE_S = 0.1

const DEFAULT_OUTPUT_PATH = joinpath(@__DIR__, "benchmarks.json")
const DEFAULT_BACKUP_PATH = joinpath(@__DIR__, "backup.json")
const JULIA_RUNNER = joinpath(@__DIR__, "julia_runner.jl")
const JAMICA_RUNNER = joinpath(@__DIR__, "jamica_runner.py")

struct BenchmarkTask
    config_key::String
    implementation::String
    config::String
    cfg::Dict{String,Any}
    runs::Int
end

include(joinpath(@__DIR__, "testsuite.jl"))

struct NvmlProcessInfo
    pid::Cuint
    usedGpuMemory::Culonglong
    gpuInstanceId::Cuint
    computeInstanceId::Cuint
end

function nvml_init()
    r = ccall((:nvmlInit_v2, libnvml), Cint, ())
    r == NVML_SUCCESS || error("nvmlInit failed: $r")
end

nvml_shutdown() = ccall((:nvmlShutdown, libnvml), Cint, ())

function nvml_device_count()
    count = Ref{Cuint}()
    r = ccall((:nvmlDeviceGetCount_v2, libnvml), Cint, (Ref{Cuint},), count)
    r == NVML_SUCCESS || error("nvmlDeviceGetCount failed: $r")
    return Int(count[])
end

function nvml_device_handle(index::Integer)
    handle = Ref{Ptr{Cvoid}}()
    r = ccall((:nvmlDeviceGetHandleByIndex_v2, libnvml), Cint, (Cuint, Ref{Ptr{Cvoid}}), Cuint(index), handle)
    r == NVML_SUCCESS || error("nvmlDeviceGetHandleByIndex failed: $r")
    return handle[]
end

function nvml_gpu_mem_for_pid(handle::Ptr{Cvoid}, pid::Integer)
    count = Ref{Cuint}(0)
    r = ccall((:nvmlDeviceGetComputeRunningProcesses_v2, libnvml),
        Cint,
        (Ptr{Cvoid}, Ref{Cuint}, Ptr{NvmlProcessInfo}),
        handle, count, C_NULL)
    if r != NVML_SUCCESS && r != NVML_ERROR_INSUFFICIENT_SIZE
        return 0
    end

    n = Int(count[])
    n > 0 || return 0

    for _ in 1:3
        buf = Vector{NvmlProcessInfo}(undef, n)
        r = ccall((:nvmlDeviceGetComputeRunningProcesses_v2, libnvml),
            Cint,
            (Ptr{Cvoid}, Ref{Cuint}, Ptr{NvmlProcessInfo}),
            handle, count, pointer(buf))
        if r == NVML_SUCCESS
            total = 0
            for i in 1:Int(count[])
                if Int(buf[i].pid) == pid && buf[i].usedGpuMemory != NVML_VALUE_NOT_AVAILABLE
                    total += Int(buf[i].usedGpuMemory)
                end
            end
            return total
        end
        r == NVML_ERROR_INSUFFICIENT_SIZE || return 0
        n = Int(count[])
        n > 0 || return 0
    end

    return 0
end

function read_rss_kib(pid::Integer)
    rss_kib = 0
    open("/proc/$pid/status", "r") do io
        for line in eachline(io)
            if startswith(line, "VmRSS:")
                rss_kib = parse(Int, split(strip(line))[2])
                break
            end
        end
    end
    return rss_kib
end

function read_child_pids(pid::Integer)
    path = "/proc/$pid/task/$pid/children"
    isfile(path) || return Int[]
    raw = try
        strip(read(path, String))
    catch
        return Int[]
    end
    isempty(raw) && return Int[]

    out = Int[]
    for tok in split(raw)
        child = tryparse(Int, tok)
        child === nothing || push!(out, child)
    end
    return out
end

function process_tree_pids(root_pid::Integer)
    seen = Set{Int}()
    stack = Int[root_pid]
    while !isempty(stack)
        pid = pop!(stack)
        pid in seen && continue
        push!(seen, pid)
        for child in read_child_pids(pid)
            child in seen || push!(stack, child)
        end
    end
    return collect(seen)
end

function parse_positive_int(name::String, value::AbstractString)
    iv = tryparse(Int, value)
    (iv === nothing || iv <= 0) && error("Invalid value for $name: '$value' (expected positive integer)")
    return iv
end

function parse_config_tokens(tokens::AbstractVector{<:AbstractString})
    length(tokens) == 7 || error(
        "Invalid config format. Expected 7 fields: dataset implementation device block_size precision threads n_iter",
    )

    dataset = lowercase(tokens[1])
    dataset in ("memorize", "small") || error("Invalid dataset '$dataset' (expected memorize or small)")
    impl = lowercase(tokens[2])
    impl in ("julia", "jamica") || error("Invalid implementation '$impl' (expected julia or jamica)")
    device = lowercase(tokens[3])
    device in ("cpu", "gpu") || error("Invalid device '$device' (expected cpu or gpu)")
    block_size = parse_positive_int("block_size", tokens[4])
    precision_raw = lowercase(tokens[5])
    precision = if precision_raw in ("float32", "f32")
        "float32"
    elseif precision_raw in ("float64", "f64")
        "float64"
    else
        error("Invalid precision '$precision_raw' (expected float32 or float64)")
    end
    threads = parse_positive_int("threads", tokens[6])
    n_iter = parse_positive_int("n_iter", tokens[7])

    config_key = join((dataset, impl, device, string(block_size), precision, string(threads), string(n_iter)), " ")
    cfg = Dict{String,Any}(
        "dataset" => dataset,
        "implementation" => impl,
        "device" => device,
        "block_size" => block_size,
        "precision" => precision,
        "threads" => threads,
        "n_iter" => n_iter,
    )
    return (config_key = config_key, implementation = impl, config = config_key, cfg = cfg)
end

function parse_single_config_arg(config_args::Vector{String})
    isempty(config_args) && return nothing
    tokens = length(config_args) == 1 ? split(strip(config_args[1])) : config_args
    return parse_config_tokens(tokens)
end

function add_suite_task!(
    tasks::Vector{BenchmarkTask},
    key_to_index::Dict{String,Int},
    config::String,
    runs::Int,
)
    runs > 0 || error("Suite task has invalid runs=$runs for config '$config'")
    parsed = parse_config_tokens(split(strip(config)))
    task = BenchmarkTask(parsed.config_key, parsed.implementation, parsed.config, parsed.cfg, runs)

    if haskey(key_to_index, task.config_key)
        idx = key_to_index[task.config_key]
        existing = tasks[idx]
        tasks[idx] = BenchmarkTask(existing.config_key, existing.implementation, existing.config, existing.cfg, max(existing.runs, runs))
    else
        push!(tasks, task)
        key_to_index[task.config_key] = length(tasks)
    end
end

function build_suite_tasks()
    tasks = BenchmarkTask[]
    key_to_index = Dict{String,Int}()
    for spec in benchmark_suite_specs()
        add_suite_task!(tasks, key_to_index, spec.config, spec.runs)
    end

    return tasks
end

const USAGE = "config format (7 fields): dataset implementation device block_size precision threads n_iter"

function parse_invocation(args::Vector{String})
    show_output = false
    runs = 1
    output_path = DEFAULT_OUTPUT_PATH
    backup_path = DEFAULT_BACKUP_PATH
    continue_on_failure = false
    config_args = String[]

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--show-output"
            show_output = true
        elseif arg == "--runs"
            i += 1
            i <= length(args) || error("--runs needs a value")
            runs = parse_positive_int("runs", args[i])
        elseif arg == "--output-file"
            i += 1
            i <= length(args) || error("--output-file needs a value")
            output_path = args[i]
        elseif arg == "--backup-file"
            i += 1
            i <= length(args) || error("--backup-file needs a value")
            backup_path = args[i]
        elseif arg == "--continue"
            continue_on_failure = true
        else
            append!(config_args, [arg])
        end
        i += 1
    end

    single = parse_single_config_arg(config_args)
    mode = single === nothing ? :suite : :single
    if mode == :suite && runs != 1
        error("--runs is only supported in single-config mode. Suite mode uses runs from testsuite.jl.")
    end

    return (
        mode = mode,
        single = single,
        show_output = show_output,
        output_path = output_path,
        backup_path = backup_path,
        runs = runs,
        continue_on_failure = continue_on_failure,
    )
end

function jamica_python()
    env = get(ENV, "AMICA_BENCH_PYTHON", "")
    isempty(env) || return env
    local_py = joinpath(@__DIR__, ".venv", "bin", "python")
    isfile(local_py) && return local_py
    py = Sys.which("python3")
    py === nothing && error("No python found for jamica runner; set AMICA_BENCH_PYTHON or run benchmark/install_cuda_machine.sh")
    return py
end

function build_command(implementation::String, config::String)
    if implementation == "julia"
        return `julia --project=$(@__DIR__) $JULIA_RUNNER $config`
    else
        return `$(jamica_python()) $JAMICA_RUNNER $config`
    end
end

function load_existing_entries(output_path::String)
    entries = Dict{String,Vector{Any}}()
    if isfile(output_path)
        raw = strip(read(output_path, String))
        if !isempty(raw)
            parsed = try
                JSON3.read(raw)
            catch err
                error("Output file is not valid JSON: $output_path\n$(sprint(showerror, err))")
            end
            parsed isa AbstractDict || error("Output file JSON must be an object/map keyed by config string: $output_path")
            for (k, v) in pairs(parsed)
                key = String(k)
                if v isa AbstractVector
                    samples = Any[]
                    for sample in v
                        sample isa AbstractDict || error("Entry array for key '$key' must contain objects")
                        push!(samples, sample)
                    end
                    entries[key] = samples
                elseif v isa AbstractDict
                    # Backward compatibility with old format: key => object
                    entries[key] = Any[v]
                else
                    error("Unsupported entry type for key '$key' in output file (expected object or array of objects)")
                end
            end
        end
    end
    return entries
end

function save_entries(output_path::String, entries::Dict{String,Vector{Any}})
    mkpath(dirname(output_path))
    open(output_path, "w") do io
        write(io, JSON3.write(entries))
        write(io, '\n')
    end
end

function move_pruned_entries_to_backup!(removed::Dict{String,Vector{Any}}, backup_path::String)
    isempty(removed) && return
    backup_entries = load_existing_entries(backup_path)
    for (key, samples) in removed
        if haskey(backup_entries, key)
            append!(backup_entries[key], samples)
        else
            backup_entries[key] = copy(samples)
        end
    end
    save_entries(backup_path, backup_entries)
end

function sanitize_output_entries!(
    output_path::String,
    backup_path::String,
    allowed_keys::Set{String},
)
    entries = load_existing_entries(output_path)
    isempty(entries) && return entries

    kept = Dict{String,Vector{Any}}()
    removed = Dict{String,Vector{Any}}()
    for (key, samples) in entries
        if key in allowed_keys
            kept[key] = samples
        else
            removed[key] = samples
        end
    end

    if !isempty(removed)
        removed_samples = sum(length(v) for v in values(removed))
        move_pruned_entries_to_backup!(removed, backup_path)
        println(
            "Pruned $(length(removed)) key(s) / $(removed_samples) samples from output and moved to backup: $(abspath(backup_path))",
        )
    end

    if length(kept) != length(entries)
        save_entries(output_path, kept)
    end
    return kept
end

function line_buffered_cmd(cmd::Cmd)
    stdbuf = Sys.which("stdbuf")
    stdbuf === nothing && return cmd
    return `$stdbuf -oL -eL $cmd`
end

function cmd_to_shell_string(cmd::Cmd)
    return join((Base.shell_escape(String(arg)) for arg in cmd.exec), " ")
end

function realtime_output_cmd(cmd::Cmd)
    buffered = line_buffered_cmd(cmd)
    script_cmd = Sys.which("script")
    script_cmd === nothing && return buffered
    return `$script_cmd -qefc $(cmd_to_shell_string(buffered)) /dev/null`
end

function poll_log_output!(
    log_path::String,
    read_offset::Base.RefValue{Int},
    show_output::Bool,
)
    isfile(log_path) || return

    file_size = try
        filesize(log_path)
    catch
        return
    end
    file_size > read_offset[] || return

    chunk = open(log_path, "r") do io
        seek(io, read_offset[])
        read(io, String)
    end
    read_offset[] = file_size
    isempty(chunk) && return

    if show_output
        print(chunk)
        flush(stdout)
    end
end

function run_single_sample(
    cmd::Cmd,
    config_key::String,
    implementation::String,
    cfg::Dict{String,Any},
    show_output::Bool,
    sample_idx::Int,
    sample_count::Int,
)

    log_path, log_io = mktemp()
    read_offset = Ref(0)

    t0 = time()
    proc = run(pipeline(realtime_output_cmd(cmd); stdout = log_io, stderr = log_io); wait = false)
    close(log_io)
    pid = Int(getpid(proc))

    nvml_ok = true
    devs = Ptr{Cvoid}[]
    try
        nvml_init()
        for i in 0:nvml_device_count()-1
            push!(devs, nvml_device_handle(i))
        end
    catch e
        nvml_ok = false
        @warn "NVML unavailable, GPU sampling disabled" exception = e
    end

    max_gpu_mib = 0
    max_rss_mib = 0

    while true
        poll_log_output!(log_path, read_offset, show_output)

        tracked_pids = process_tree_pids(pid)

        rss_kib = 0
        for tracked_pid in tracked_pids
            rss_kib += try
                read_rss_kib(tracked_pid)
            catch
                0
            end
        end
        max_rss_mib = max(max_rss_mib, rss_kib ÷ 1024)

        if nvml_ok
            total_bytes = 0
            for tracked_pid in tracked_pids
                for d in devs
                    total_bytes += nvml_gpu_mem_for_pid(d, tracked_pid)
                end
            end
            max_gpu_mib = max(max_gpu_mib, total_bytes ÷ (1024 * 1024))
        end

        !process_running(proc) && break
        sleep(SAMPLE_S)
    end

    wait(proc)
    poll_log_output!(log_path, read_offset, show_output)

    t_done = time()
    runtime = t_done - t0

    if nvml_ok
        nvml_shutdown()
    end

    rm(log_path; force = true)

    println("\n========== Benchmark Summary ==========")
    println("Sample: $sample_idx/$sample_count")
    println("Key: $config_key")
    println("Implementation: $implementation")
    println("Command: " * join(cmd.exec, " "))
    println("Root PID: $pid")
    println("Runtime (incl. startup + warm-up): $(round(runtime; digits = 3)) s")
    println("Max CPU RSS: $max_rss_mib MiB")
    println("Max GPU memory: " * (nvml_ok ? "$max_gpu_mib MiB" : "unavailable"))
    println("=======================================")

    if !success(proc)
        println("Run failed; not recording benchmark entry for key: $config_key")
        return nothing
    end

    return Dict(
        "key" => config_key,
        "implementation" => implementation,
        "params" => cfg,
        "command" => join(cmd.exec, " "),
        "pid" => pid,
        "pid_scope" => "process_tree",
        "sample_index" => sample_idx,
        "sample_count" => sample_count,
        "runtime_s" => round(runtime; digits = 6),
        "max_cpu_rss_mib" => max_rss_mib,
        "max_gpu_memory_mib" => nvml_ok ? max_gpu_mib : nothing,
    )
end

function run_config_samples(
    config_key::String,
    implementation::String,
    config::String,
    cfg::Dict{String,Any},
    show_output::Bool,
    runs::Int,
)
    cmd = build_command(implementation, config)
    samples = Vector{Any}()
    runtime_samples = Float64[]
    cpu_samples = Float64[]
    gpu_samples = Float64[]

    for i in 1:runs
        println("Running sample $i/$runs for key: $config_key")
        entry = run_single_sample(cmd, config_key, implementation, cfg, show_output, i, runs)
        entry === nothing && return nothing
        push!(samples, entry)

        runtime_val = try
            Float64(entry["runtime_s"])
        catch
            nothing
        end
        runtime_val === nothing || push!(runtime_samples, runtime_val)

        cpu_val = try
            Float64(entry["max_cpu_rss_mib"])
        catch
            nothing
        end
        cpu_val === nothing || push!(cpu_samples, cpu_val)

        gpu_val = try
            gpu_raw = entry["max_gpu_memory_mib"]
            gpu_raw === nothing ? nothing : Float64(gpu_raw)
        catch
            nothing
        end
        gpu_val === nothing || push!(gpu_samples, gpu_val)
    end

    println("\n========== Benchmark Median (across $runs samples) ==========")
    println("Key: $config_key")
    !isempty(runtime_samples) && println("Median runtime: $(round(median(runtime_samples); digits = 6)) s")
    !isempty(cpu_samples) && println("Median max CPU RSS: $(round(median(cpu_samples); digits = 3)) MiB")
    !isempty(gpu_samples) && println("Median max GPU memory: $(round(median(gpu_samples); digits = 3)) MiB")
    println("==============================================================")

    return samples
end

function run_single_mode!(
    single::NamedTuple,
    runs::Int,
    show_output::Bool,
    output_path::String,
    entries::Dict{String,Vector{Any}},
)
    samples = run_config_samples(single.config_key, single.implementation, single.config, single.cfg, show_output, runs)
    samples === nothing && return 1
    entries[single.config_key] = samples
    save_entries(output_path, entries)
    return 0
end

mutable struct SuiteRunReport
    total::Int
    succeeded::Int
    skipped::Int
    failed::Vector{String}
    interrupted::Bool
end

function print_suite_report(report::SuiteRunReport)
    processed = report.succeeded + report.skipped + length(report.failed)
    println("\n========== Suite Report ==========")
    println("Total tasks: $(report.total)")
    println("Processed: $processed")
    println("Succeeded: $(report.succeeded)")
    println("Skipped: $(report.skipped)")
    println("Failed: $(length(report.failed))")
    println("Interrupted: " * (report.interrupted ? "yes" : "no"))
    if !isempty(report.failed)
        println("Failed keys:")
        for key in report.failed
            println(" - $key")
        end
    end
    println("==================================")
end

function split_preexisting_suite_tasks(
    tasks::Vector{BenchmarkTask},
    entries::Dict{String,Vector{Any}},
)
    runnable = NamedTuple{(:task, :existing_count),Tuple{BenchmarkTask,Int}}[]
    skipped = 0
    for task in tasks
        existing_count = haskey(entries, task.config_key) ? length(entries[task.config_key]) : 0
        if existing_count >= task.runs
            skipped += 1
        else
            push!(runnable, (task = task, existing_count = existing_count))
        end
    end
    return runnable, skipped
end

function run_suite_mode!(
    tasks::Vector{BenchmarkTask},
    show_output::Bool,
    output_path::String,
    entries::Dict{String,Vector{Any}},
    continue_on_failure::Bool,
)
    runnable, preexisting_skipped = split_preexisting_suite_tasks(tasks, entries)
    report = SuiteRunReport(length(tasks), 0, preexisting_skipped, String[], false)
    exit_code = 0

    if preexisting_skipped > 0
        println("Filtered pre-existing benchmarks: $preexisting_skipped/$(report.total)")
    end

    try
        for (run_idx, item) in enumerate(runnable)
            task = item.task
            progress_idx = preexisting_skipped + run_idx

            if item.existing_count > 0
                println("[$progress_idx/$(report.total)] Re-running benchmark: $(task.config_key) ($(item.existing_count)/$(task.runs) samples present)")
            else
                println("[$progress_idx/$(report.total)] Running benchmark ($(task.runs) samples): $(task.config_key)")
            end

            samples = run_config_samples(task.config_key, task.implementation, task.config, task.cfg, show_output, task.runs)
            if samples === nothing
                push!(report.failed, task.config_key)
                println("[$progress_idx/$(report.total)] FAILED: $(task.config_key)")
                if !continue_on_failure
                    exit_code = 1
                    break
                end
                continue
            end

            entries[task.config_key] = samples
            save_entries(output_path, entries)
            report.succeeded += 1
        end
    catch err
        if err isa InterruptException
            report.interrupted = true
            exit_code = 130
        else
            rethrow()
        end
    end

    print_suite_report(report)

    if exit_code != 0
        return exit_code
    end
    return isempty(report.failed) ? 0 : 1
end

function main(args::Vector{String})
    invocation = parse_invocation(args)
    suite_tasks = build_suite_tasks()
    allowed_keys = Set(task.config_key for task in suite_tasks)
    if invocation.mode == :single
        push!(allowed_keys, invocation.single.config_key)
    end

    entries = sanitize_output_entries!(invocation.output_path, invocation.backup_path, allowed_keys)

    if invocation.mode == :single
        return run_single_mode!(invocation.single, invocation.runs, invocation.show_output, invocation.output_path, entries)
    else
        return run_suite_mode!(suite_tasks, invocation.show_output, invocation.output_path, entries, invocation.continue_on_failure)
    end
end

exit(main(ARGS))
