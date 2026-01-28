using Test
using Statistics
using Amica
using TimerOutputs

include("util.jl")

data = Float64.(read_fdt("input/Memorize.fdt"; ncols=71, T=Float32))'

(N, n) = size(data)

lrate = Amica.LearningRate{Float64}(newtrate=Float64(1.0))

myAmica = SingleModelAmica(Float64, ncomps=n, nsamples=N, m=3, ArrayType=Array)
Amica.amica!(myAmica, data, maxiter=1, newt_start_iter=0, lrate=lrate)