#s = sin.(t * collect(0.5:0.8:pi)')'#rand(10,10000)
using Revise
using SignalAnalysis
using Amica
includet("/scratch/projects/fapra_amica/src/simulate_data.jl")
includet("/scratch/projects/fapra_amica/src/fortran_tools.jl")
n_chan = 4
n_time = 1000
x, _A, s = simulate_data(; T=50, n_chan, n_time, type=:gg6)
i_scale, i_location, i_A = init_params(; n_chan, n_time)

#s[3, :] = sin.(10 .* t)
#s[4, :] = cos.(20 .* t)
#s = s .* [1,2,3,4]
A = rand(size(s, 1), size(s, 1))
A = [1 1 0 0; 0 1 1 0; 0 0 1 1; 1 0 1 0]
#A = [1 2; 0.5 -3]
x = A * s

#A = [1 1 0 1; 1 1 0 0; 1 0 1 1; 0 0 0 1]
#x = hcat(x,A*s) 
am = Amica.fit(SingleModelAmica, x; maxiter=50, do_sphering=true)
#amm = fit(MultiModelAmica,x;maxiter=500,M=1)
size(am.A)

#---
W = pinv(am.A)
W = pinv(am.A) * am.S
#W = pinv(pinv(am.S) * am.A)*(am.S)
#W = pinv(pinv(am.S) * am.A * am.S)
using CairoMakie

f = Figure()
series(f[1, 1], s[:, 1:500])
ax, h = heatmap(f[1, 2], A)
Colorbar(f[1, 3], h)


series(f[2, 1], (W*x)[:, 1:500])
ax, h = heatmap(f[2, 2], am.A[:, :, 1])
Colorbar(f[2, 3], h)

series(f[4, 1], s[:, 1:50])
series(f[4, 2], (W*x)[:, 1:50])
f
#series(f[4,1],inv(W)'*x)
#series(f[6,1],W'*x)

#----
using PyMNE
data_path = pyconvert(String, @py(str(PyMNE.datasets.ssvep.data_path())))
bids_fname = joinpath(data_path, "sub-02", "ses-01", "eeg", "sub-02_ses-01_task-ssvep_eeg.vhdr")


raw = PyMNE.io.read_raw_brainvision(bids_fname, preload=true, verbose=false)
raw.resample(128)
raw.filter(l_freq=1, h_freq=nothing, fir_design="firwin")
d = pyconvert(Array, raw.get_data(; units="uV"))


am = fit(MultiModelAmica, d; maxiter=500, M=2)

#----
raw_memory = PyMNE.io.read_epochs_eeglab("/data/export/users/ehinger/amica_recompile/amica/Memorize.set")
d_memory = pyconvert(Array, raw_memory.get_data(; units="uV"))

d_memory = reshape(permutedims(d_memory, (2, 3, 1)), 71, :)

am2 = fit(SingleModelAmica, d_memory, M=1)