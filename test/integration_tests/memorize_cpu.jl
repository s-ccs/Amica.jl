using Test
using Statistics
using Amica

include("util.jl")

@testset "run against memorize set" begin
    data = Float64.(read_fdt("input/Memorize.fdt"; ncols=71, T=Float32))'
    LL = (read_fdt("LL_newton"; ncols=1, T=Float64))'

    (N, n) = size(data)

    # myAmica = fit(Float32, SingleModelAmica, data; maxiter=5, do_sphering=true, remove_mean=true, m=3,)
    lrate = Amica.LearningRate{Float64}(newtrate=Float64(1.0))
    myAmica = SingleModelAmica(Float64, ncomps=n, nsamples=N, m=3, num_threads=1, block_size=2000)
    Amica.amica!(myAmica, data, maxiter=5, newt_start_iter=0, lrate=lrate, show_timing=true)


end