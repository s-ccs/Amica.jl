ENV["MATLAB_ROOT"] = "/opt/common/apps/matlab/r2021a/"

using MATLAB
using Amica
using SignalAnalysis
using LinearAlgebra
using Revise 
#---
using CairoMakie
#---
includet("/scratch/projects/fapra_amica/src/simulate_data.jl")
includet("/scratch/projects/fapra_amica/src/fortran_tools.jl")
n_chan=10
n_time=500_000
x,A,s = simulate_data(;T=50,n_chan,n_time,type=:gg6)
i_scale,i_location,i_A = init_params(;n_chan,n_time)

#---
f  = Figure()
series(f[1,1],s,axis=(;title="source"))
#xlims!(1,20)
series(f[2,1],x,axis=(;title="mixed"))
f

#---

maxiter = 50


# matlab run
mat"""
tic
[mA,mc,mLL,mLt,mgm,malpha,mmu,mbeta,mrho] = amica_a($x,1,3,$maxiter,$i_location,$i_scale,$i_A,0);
mt = toc
"""
mat"""
tic
[mAopt,mWopt,mSopt,mkhindsopt,mcopt,mLLopt,mLtopt,mgmopt,malphaopt,mmuopt,mbetaopt,mrhoopt] = amica_optimized($x,1,3,$maxiter,1,1,$i_location,$i_scale,$i_A);
mtopt = toc

"""
mA = @mget(mA); # get matlab var
mAopt = @mget(mAopt); # get matlab opt var

t_mopt = @mget(mtopt); # get matlab var
t_m    = @mget(mt); # get matlab var

# Fortran setup + ran
fortran_setup(Float32.(x);max_threads=1,max_iter=maxiter)
t_f32 = @elapsed run(`/scratch/projects/fapra_amica/fortran/amica15test julia.param`)
fA = reshape(reinterpret(Float64,(read("amicaout/A"))),n_chan,n_chan)

fortran_setup(Float64.(x);max_threads=1,max_iter=maxiter,dble_data=1)
t_f64 = @elapsed run(`/scratch/projects/fapra_amica/fortran/amica15test julia.param`)

# Julia run
am32 = SingleModelAmica(Float32.(x);maxiter=maxiter,A=deepcopy(i_A),location=deepcopy(i_location),scale=deepcopy(i_scale))
t_am32 = @elapsed fit!(am32,Float32.(x))

am64 = SingleModelAmica(Float64.(x);maxiter=maxiter,A=deepcopy(i_A),location=deepcopy(i_location),scale=deepcopy(i_scale))
t_am64 = @elapsed fit!(am64,Float64.(x))
#am = SingleModelAmica(Float16.(x);maxiter=maxiter,A=i_A,location=i_location,scale=i_scale)
#@time fit!(am,x)
#vcat(@mget(mLL),am.LL')

#---
f2 = Figure(size=(800,800))
series(f2[1,1],inv(am64.A)*x, axis=(;title="unmixed julia64"))
series(f2[2,1],inv(am32.A)*x, axis=(;title="unmixed julia32"))
series(f2[3,1],inv(mA)*x, axis=(;title="unmixed matlab"))
series(f2[4,1],inv(mAopt)*x, axis=(;title="unmixed matlab_optimizd"))
series(f2[5,1],inv(fA)*x, axis=(;title="unmixed fortran"))
series(f2[6,1],s,axis=(;title="original source"))
series(f2[7,1],x, axis=(;title="original mixed"))
hidedecorations!.(f2.content)

linkxaxes!(f2.content...)
xlims!(0,100)
f2


#--- compare LLs

visibnoise = 0.02 # add little bit of noise to distinguish identical lines 
mLL = @mget mLL
mLLopt = @mget mLLopt
fLL = reinterpret(Float64,(read("amicaout/LL")))

f = Figure(size=(1024,512))
ax = f[1,1] = Axis(f)
labels = ["julia64","julia32","matlab", "matlab opt","fortran"]
for (ix,d) = enumerate([am64.LL, am32.LL,mLL[1,:],mLLopt[1,:],fLL])
    lines!(ax,d .+ rand(size(d)...).*visibnoise;label=labels[ix])
end
axislegend(ax)
ylims!(ax,-5,0.1)


ax_T = f[1,2] = Axis(f)
scatter!(ax_T,[t_am32,t_am64,t_m,t_mopt, t_f32,t_f64])
ax_T.xticks = 1:6
ax_T.xtickformat = x->["am32","am64","mat","matopt","f32","f64"][Int.(x)]

f


#---


#----
am32 = SingleModelAmica(Float32.(x);maxiter=10)#,A=deepcopy(i_A),location=deepcopy(i_location),scale=deepcopy(i_scale))
@profview fit!(am32,Float32.(x))

fortran_setup(Float32.(x);max_threads=1,max_iter=5)
t_f32 = @elapsed run(`/scratch/projects/fapra_amica/fortran/amica15test julia.param`)