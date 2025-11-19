using Revise
using Amica
using BenchmarkTools
using Metal

include("util.jl")

data = Float32.(read_fdt("input/Memorize.fdt"; ncols=71, T=Float32));

gpuAmica = SingleModelAmica(data; m=3, ArrayType=MtlArray);
myAmica = SingleModelAmica(data; m=3, ArrayType=Array);

@btime Metal.@sync Amica.update_y_rho!(myAmica);
@btime Metal.@sync Amica.update_y_rho!(gpuAmica);

@btime Metal.@sync Amica.calculate_Q!(myAmica);
@btime Metal.@sync Amica.calculate_Q!(gpuAmica);

@btime Metal.@sync Amica.calculate_u_and_Lt!(myAmica);
@btime Metal.@sync Amica.calculate_u_and_Lt!(gpuAmica);