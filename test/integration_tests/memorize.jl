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
    myAmica = SingleModelAmica(Float32, ncomps=n, nsamples=N, m=3, ArrayType=CuArray)
    Amica.amica!(myAmica, data, maxiter=40)

    using Plots
    gr()   # Backend aktivieren

    p = plot(1:40, LL[1:40, :], label="Expected LL", marker=:circle)
    plot!(p, 1:40, myAmica.LL[1:40], label="Amica LL", marker=:circle)
    savefig(p, "out.png")
    # @info myAmica.LL
end