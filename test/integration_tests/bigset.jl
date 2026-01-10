using Test
using Statistics
using Amica
using CUDA


using PyMNE;
raw = PyMNE.io.read_raw_fif("/home/fapra_morlock/hallo/Amica.jl/test/integration_tests/sub-030_ses-001_task-Default_run-1_proc-filt_raw.fif")
data = Float32.(pyconvert(Array, raw.get_data())')

write("big.bin", x)


include("util.jl")

(N, n) = size(data)

lrate = Amica.LearningRate{Float32}(newtrate=Float32(1.0))
myAmica = SingleModelAmica(Float32, ncomps=n, nsamples=N, m=3, ArrayType=CuArray)
Amica.amica!(myAmica, data, maxiter=40, newt_start_iter=0, lrate=lrate)

using Plots
gr()   # Backend aktivieren

p = plot(1:40, LL[1:40, :], label="Expected LL", marker=:circle)
plot!(p, 1:40, myAmica.LL[1:40], label="Amica LL", marker=:circle)
savefig(p, "out.png")
# @info myAmica.LL