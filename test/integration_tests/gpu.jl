using Revise
using Amica
using BenchmarkTools
using Metal
using LinearAlgebra

include("util.jl")

data = Float32.(read_fdt("input/Memorize.fdt"; ncols=71, T=Float32))';

(N, n) = size(data)
lrate = Amica.LearningRate{Float32}()

# myAmica = SingleModelAmica(Float32, ncomps=n, nsamples=N, m=3, ArrayType=Array);

myAmica = SingleModelAmica(Float32, ncomps=n, nsamples=N, m=3, ArrayType=MtlArray);


Amica.initialize_shape_parameter!(myAmica, lrate)
Amica.removeMean!(data)
S = Amica.sphering!(data)
myAmica.S = S
myAmica.LLdetS = logabsdet(S |> Array)[1]

Amica.update_sources!(myAmica, data)

Amica.calculate_y!(myAmica)
Amica.update_y_rho!(myAmica);

Amica.calculate_u_and_Lt!(myAmica);

Amica.update_parameters!(myAmica, lrate, true, true)