using MAT
using Amica
using SignalAnalysis
using LinearAlgebra
using PyMNE

#_____________________________________________
# #Generate multiple sin mixed with pink gaussian
# t = range(0,20*π,length=100)
# s =rand(PinkGaussian(length(t)),4)'
# s[2,:] = sin.(t)
# s[3,:] = sin.(2 .* t)
# s[4,:] = sin.(10 .* t)
# s = s .* [1,2,3,4]
# #A = rand(size(s,1),size(s,1))
# A = [1 1 0 0; 0 1 1 0; 0 0 1 1; 1 0 1 0]
# x = A*s
# #_____________________________________________
#_____________________________________________
#get eeg data
data_path = pyconvert(String,@py(str(PyMNE.datasets.ssvep.data_path())))
bids_fname =  joinpath(data_path,"sub-02","ses-01","eeg","sub-02_ses-01_task-ssvep_eeg.vhdr")


raw = PyMNE.io.read_raw_brainvision(bids_fname, preload=true, verbose=false)
raw.resample(128)
raw.filter(l_freq=1, h_freq=nothing, fir_design="firwin")
x = pyconvert(Array,raw.get_data(;units="uV"))
#_____________________________________________

m = 3
M = 1
n = 32

#initialise random parameters (A, beta, mu) before saving them
beta = ones(m, n, M) + 0.1 * randn(m, n, M)

if m > 1
    mu = 0.1 * randn(m, n, M)
else
    mu = zeros(m, n, M)
end

eye = Matrix{Float64}(I, n, n)
A_init = zeros(n,n,M)
for h in 1:M
    A_init[:,:,h] = eye[n] .+ 0.1*rand(n,n)
    for i in 1:n
        A_init[:,i,h] = A_init[:,i,h] / norm(A_init[:,i,h])
    end
end

file = matopen("test/eeg_data2.mat", "w")
write(file, "x", x)
write(file, "s", s) #only save for self-mixed data
write(file, "A", A) # "

write(file, "beta_init", beta)
write(file, "A_init", A_init)
write(file, "mu_init", mu)
close(file)

# file = matopen("pink_sinus_data.mat")
# x = read(file, "data")
# close(file)