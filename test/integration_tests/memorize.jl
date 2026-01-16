using Test
using Statistics
using Amica
using CUDA

include("util.jl")

@testset "run against memorize set" begin
    data = (read_fdt("input/Memorize.fdt"; ncols=71, T=Float32))'
    LL = (read_fdt("LL"; ncols=1, T=Float64))'

    (N, n) = size(data)

    # myAmica = fit(Float32, SingleModelAmica, data; maxiter=5, do_sphering=true, remove_mean=true, m=3, Array=CuArray)
    lrate = Amica.LearningRate{Float32}(newtrate=Float32(1.0))
    myAmica = SingleModelAmica(Float32, ncomps=n, nsamples=N, m=3, ArrayType=CuArray)
    Amica.amica!(myAmica, data, maxiter=500, newt_start_iter=50, lrate=lrate)

    # @info myAmica.LL
end