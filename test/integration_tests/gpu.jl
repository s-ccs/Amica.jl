using Revise
using Amica
using BenchmarkTools
using Metal

include("util.jl")

data = Float32.(read_fdt("input/Memorize.fdt"; ncols=71, T=Float32));


(N, n) = size(data)

myAmica = SingleModelAmica(Float32, ncomps=n, nsamples=N, m=3, ArrayType=Array);

gpuAmica = SingleModelAmica(Float32, ncomps=n, nsamples=N, m=3, ArrayType=MtlArray);

Metal.@sync Amica.calculate_y!(myAmica)
Metal.@sync Amica.calculate_y!(gpuAmica)

Metal.@sync Amica.update_y_rho!(myAmica);
Metal.@sync Amica.update_y_rho!(gpuAmica);

@btime Metal.@sync Amica.calculate_u_and_Lt!(myAmica);
@btime Metal.@sync Amica.calculate_u_and_Lt!(gpuAmica);
