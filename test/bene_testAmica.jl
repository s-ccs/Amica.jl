#s = sin.(t * collect(0.5:0.8:pi)')'#rand(10,10000)
using SignalAnalysis
using Amica

t = range(0,20*π,length=10000)
s =rand(PinkGaussian(length(t)),20)'
s[2,:] = sin.(t)
s[3,:] = sin.(2 .* t)
s[4,:] = sin.(10 .* t)
#s = s .* [1,2,3,4]
A = rand(size(s,1),size(s,1))
A = [1 1 0 0; 0 1 1 0; 0 0 1 1; 1 0 1 0]

x = A*s

#A = [1 1 0 1; 1 1 0 0; 1 0 1 1; 0 0 0 1]
#x = hcat(x,A*s) 
am = fit(SingleModelAmica,x;maxiter=500)
amm = fit(MultiModelAmica,x;maxiter=500,M=1)
size(am.A)
W = inv(am.A[:,:,1]) #previously [:,:,2]


#---
using CairoMakie

f = Figure()
series(f[1,1],s[:,1:500])
ax,h = heatmap(f[1,2],A)
Colorbar(f[1,3],h)


series(f[2,1],W*x)
ax,h = heatmap(f[2,2],am.A[:,:,1])
Colorbar(f[2,3],h)

series(f[4,1],x[:,1:500])
series(f[4,2],(W*x)[:,1:500])
f
#series(f[4,1],inv(W)'*x)
#series(f[6,1],W'*x)

#----
using PyMNE
data_path = pyconvert(String,@py(str(PyMNE.datasets.ssvep.data_path())))
bids_fname =  joinpath(data_path,"sub-02","ses-01","eeg","sub-02_ses-01_task-ssvep_eeg.vhdr")


raw = PyMNE.io.read_raw_brainvision(bids_fname, preload=true, verbose=false)
raw.resample(128)
raw.filter(l_freq=1, h_freq=nothing, fir_design="firwin")
d = pyconvert(Array,raw.get_data(;units="uV"))


am = fit(MultiModelAmica,d;maxiter=500,M=2)

#----
raw_memory = PyMNE.io.read_epochs_eeglab("/data/export/users/ehinger/amica_recompile/amica/Memorize.set")
d_memory = pyconvert(Array,raw_memory.get_data(;units="uV"))

d_memory = reshape(permutedims(d_memory,(2,3,1)),71,:)

am2 = fit(SingleModelAmica,d_memory,M=1)