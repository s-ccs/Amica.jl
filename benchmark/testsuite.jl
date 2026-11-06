function benchmark_suite_specs()
    specs = NamedTuple{(:config, :runs),Tuple{String,Int}}[]
    add_spec(config::String, runs::Int) = push!(specs, (config=config, runs=runs))

    datasets = ("small", "memorize")   # small: 172704×19, memorize: 319500×71
    n_iter = 40
    runs = 3

    for dataset in datasets
        # CPU: float64, julia with 1 and 16 threads (jamica threads via XLA, runs at 1)
        # block 10_000: >= 16 blocks available even for small
        add_spec("$dataset julia cpu 10000 float64 1 $n_iter", runs)
        add_spec("$dataset julia cpu 10000 float64 16 $n_iter", runs)
        add_spec("$dataset jamica cpu 10000 float64 1 $n_iter", runs)

        # GPU: block 50_000 (> 10_000 performs well per docs), no CPU multithreading
        for precision in ("float32", "float64"), implementation in ("julia", "jamica")
            add_spec("$dataset $implementation gpu 50000 $precision 1 $n_iter", runs)
        end
    end

    return specs
end
