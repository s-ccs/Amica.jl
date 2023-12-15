using MATLAB
using Amica
using SignalAnalysis
using LinearAlgebra
using Revise 
using CairoMakie
#---
includet("src/simulate_data.jl")
includet("src/fortran_tools.jl")
n_chan=3
n_time=5000
x,A,s = simulate_data(;T=50,n_chan,n_time,type=:gg6)
i_scale,i_location,i_A = init_params(;n_chan,n_time)

f  = Figure()
series(f[1,1],s,axis=(;title="source"))
#xlims!(1,20)
series(f[2,1],x,axis=(;title="mixed"))
f

#---

maxiter = 500

# matlab run
mat"""
[mA,mc,mLL,mLt,mgm,malpha,mmu,mbeta,mrho] = amica_a($x,1,3,$maxiter,$i_location,$i_scale,$i_A,0);
"""
mat"""
[mAopt,mWopt,mSopt,mkhindsopt,mcopt,mLLopt,mLtopt,mgmopt,malphaopt,mmuopt,mbetaopt,mrhoopt] = amica_optimized($x,1,3,$maxiter,1,1,$i_location,$i_scale,$i_A);
"""
mA = @mget(mA); # get matlab var
mAopt = @mget(mAopt); # get matlab opt var

# Fortran setup + ran
fortran_setup(x;max_threads=1,max_iter=maxiter)
run(`/scratch/projects/fapra_amica/fortran/amica15test julia.param`)
fA = reshape(reinterpret(Float64,(read("amicaout/A"))),n_chan,n_chan)

# Julia run
am = SingleModelAmica(x;maxiter=maxiter,A=i_A,location=i_location,scale=i_scale)
fit!(am,x)
#vcat(@mget(mLL),am.LL')

#---
f2 = Figure(size=(800,800))
series(f2[1,1],inv(am.A)*x, axis=(;title="unmixed julia"))
series(f2[2,1],inv(mA)*x, axis=(;title="unmixed matlab"))
series(f2[3,1],inv(mAopt)*x, axis=(;title="unmixed matlab_optimizd"))
series(f2[4,1],.-inv(fA')*x, axis=(;title="unmixed fortran"))
series(f2[5,1],s,axis=(;title="original source"))
series(f2[6,1],x, axis=(;title="original mixed"))
hidedecorations!.(f2.content)

linkxaxes!(f2.content...)
xlims!(0,100)
f2


#--- compare LLs
LL = am.LL
mLL = @mget mLL
mLLopt = @mget mLLopt
fLL = reinterpret(Float64,(read("amicaout/LL")))
f = Figure()
ax = f[1,1] = Axis(f)
labels = ["julia","matlab", "matlab opt","fortran"]
for (ix,d) = enumerate([LL,mLL[1,:],mLLopt[1,:],fLL])
    lines!(ax,d;label=labels[ix])
end
axislegend(ax)
ylims!(ax,-1.3,-1)
f

#--- compare A
