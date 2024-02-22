using MATLAB: mxSINGLE_CLASS
ENV["MATLAB_ROOT"] = "/opt/common/apps/matlab/r2021a/"
using Revise

using MATLAB
using Amica
using SignalAnalysis
using LinearAlgebra
#---
using CairoMakie
#---
includet("/scratch/projects/fapra_amica/src/simulate_data.jl")
includet("/scratch/projects/fapra_amica/src/fortran_tools.jl")
n_chan = 4
n_time = 10_000#60_000
x, _A, s = simulate_data(; T=50, n_chan, n_time, type=:gg6)
i_scale, i_location, i_A = init_params(; n_chan, n_time)

#---
f = Figure()
series(f[1, 1], s[:, 1:100], axis=(; title="source"))
#xlims!(1,20)
series(f[2, 1], x[:, 1:100], axis=(; title="mixed"))
f

#----
maxiter = 50
fortran_setup(Float32.(x); max_threads=10, max_iter=maxiter, dble_data=0)
run(`/scratch/projects/fapra_amica/fortran/amica15test julia.param`)

mat"""
tic
[mA,mc,mLL,mLt,mgm,malpha,mmu,mbeta,mrho] = amica_a($x,1,3,$maxiter,$i_location,$i_scale,$i_A,0);
mt = toc
"""

# Julia run
am = SingleModelAmica(x; maxiter=maxiter)#, A=deepcopy(i_A), location=deepcopy(i_location), scale=deepcopy(i_scale))
fit!(am, x; do_sphering=true)

#---
fLL = reinterpret(Float64, (read("amicaout/LL")))
scatter(vec(@mget(mLL)), label="matlab")
scatter!(am.LL, label="julia")
scatter!(fLL, label="fortran")
ylims!(-2, 0)
axislegend()
current_figure()

#---
fA = reshape(reinterpret(Float64, (read("amicaout/A"))), n_chan, n_chan)
fS = reshape(reinterpret(Float64, (read("amicaout/S"))), n_chan, n_chan)


f2 = Figure()#size=(800, 800))
#series(f2[1, 1], (inv(a.A) * x), axis=(; title="unmixed julia64"))
series(f2[end+1, 1], inv(am.A) * am.S * x[:, 1:100], axis=(; title="unmixed julia32"))
#series(f2[end+1, 1], inv(mA) * x, axis=(; title="unmixed matlab"))
#series(f2[end+1, 1], inv(mAopt) * x, axis=(; title="unmixed matlab_optimizd"))
series(f2[end+1, 1], (inv(fA)*fS*x)[:, 1:100], axis=(; title="unmixed fortran"))
series(f2[end+1, 1], s[:, 1:100], axis=(; title="original source"))
series(f2[end+1, 1], x[:, 1:100], axis=(; title="original mixed"))
hidedecorations!.(f2.content)
f2

#---
xc = Amica.CuArray(Float32.(x))
amc = Amica.fit(CuSingleModelAmica, xc; maxiter=3)


#---
import IntelVectorMath as IVM
function A(myAmica)
    IVM.abs!(myAmica.y_rho, myAmica.y)
    for i in 1:size(myAmica.y_rho, 2)
        for j in 1:size(myAmica.y_rho, 1)
            @views _y_rho = myAmica.y_rho[j, i, :]
            Amica.optimized_pow!(_y_rho, _y_rho, myAmica.learnedParameters.shape[j, i])
        end
    end

end

CUDA.@profile A(amc)
