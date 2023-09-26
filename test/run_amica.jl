using Amica
#using CairoMakie
using MAT
using LinearAlgebra

#___________________________________________________________________________________
#Sinus data from mat file
#file = matopen("test/eeg_data.mat")
file = matopen("test/testdata/pink_sinus_data.mat")
x = read(file, "x")
s = read(file, "s")
A = read(file, "A")

beta_init = read(file, "beta_init")
mu_init = read(file, "mu_init")
A_init = read(file, "A_init")

close(file)
@time am = fit(SingleModelAmica,x;maxiter=100, do_sphering = true,remove_mean = true,m=3, scale=beta_init[:,:,1], location=mu_init[:,:,1], A=copy(A_init[:,:,1]))
#@time am = fit(MultiModelAmica,x;maxiter=100, m=3,M = 2,scale=beta_init, location=mu_init, A=copy(A_init), remove_mean = true)

#Plots original data, mixed data, unmixed data and likelihood over iterations. Requires CairoMakie.jl
#W = pinv(am.models[1].A);
#---
# f = Figure()
# series(f[1,1],s[:,1:100])
# #ax,h = heatmap(f[1,2],A)
# #Colorbar(f[1,3],h)

# series(f[2,1],x[:,1:100])
# #ax,h = heatmap(f[2,2],am.A[:,:,1])
# #Colorbar(f[2,3],h)

# series(f[3,1],pinv(am.A[:,:,1])*x[:,1:100])
# #series(f[4,1],am.Lt)
# #series(f[4,1],am.LL)

# #series(f[4,2],(W*x)[:,1:100])

# f
# #series(f[4,1],pinv(W)'*x)
# #series(f[6,1],W'*x)

# #----