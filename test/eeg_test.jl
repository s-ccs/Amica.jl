using CairoMakie
using SignalAnalysis
using Amica

using PyMNE
data_path = pyconvert(String,@py(str(PyMNE.datasets.ssvep.data_path())))
bids_fname =  joinpath(data_path,"sub-02","ses-01","eeg","sub-02_ses-01_task-ssvep_eeg.vhdr")


raw = PyMNE.io.read_raw_brainvision(bids_fname, preload=true, verbose=false)
raw.resample(128)
raw.filter(l_freq=1, h_freq=nothing, fir_design="firwin")
d = pyconvert(Array,raw.get_data(;units="uV"))

am = fit(MultiModelAmica,d;maxiter=500,M=1)

f = Figure()
#series(f[1,1],d)
ax,h = heatmap(f[1,2],am.A)
Colorbar(f[1,3],h)


series(f[2,1],W*x)
ax,h = heatmap(f[2,2],am.A[:,:,1])
Colorbar(f[2,3],h)

series(f[4,1],x[:,1:500])
series(f[4,2],(W*x)[:,1:500])

series(f[3,2],am.Lt)
series(f[3,1],am.LL)
f