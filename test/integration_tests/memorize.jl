using Test
using Statistics
using Amica

include("util.jl")

@testset "run against memorize set" begin
    # verify the raw data is identical
    data = Float64.(read_fdt("input/Memorize.fdt"; ncols=71, T=Float32))

    myAmica = fit(SingleModelAmica, data; maxiter=5, do_sphering=true, remove_mean=true, m=3)

    # @info myAmica.LL

end
