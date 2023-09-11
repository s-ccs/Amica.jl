using Amica
#using CairoMakie
using MAT
using LinearAlgebra

# #Old hardcoded stuff
# data = [1 4; 4 1]*Float64.([1.0 2 3; 4 5 6])
# M = 2 #number of mixture models
# m = 3 #number of source density mixtures


# A = zeros(size(data,1),size(data,1),M)
# A[:,:,1] = [1.0 0.003; -0.05 1.0]
# A[:,:,2] = [2.0 0.003; -0.05 1.0]

# beta = ones(m, size(data,1), M)
# beta[:,:,1] = [1.1 0.9; 1.0 0.9; 0.9 0.8]
# beta[:,:,2] = [1.2 0.9; 1.1 0.8; 0.9 0.7]

# mu = zeros(m, size(data,1), M)
# mu[:,:,1] = [0.1 0.9; -0.01 0.0; 0.0 -0.02] #todo: wieder rnd mu einfügen
# mu[:,:,2] = [0.2 1; -0.01 0.0; 0.0 -0.03]

# amica = MultiModelAmica(Amica.removeMean!(data); M=2, maxiter=4, A=A ,mu=mu, beta=beta)
# Amica.amica!(amica, data; mindll=1e-8, iterwin=1)

#fit(amica, x, )


# z, A, Lt, LL = amica(data, M, m, maxiter, update_rho, mindll, iterwin, do_newton, remove_mean)

#___________________________________________________________________________________
#Sinus data from mat file
#file = matopen("test/eeg_data.mat")
file = matopen("test/pink_sinus_data.mat")
x = read(file, "x")
s = read(file, "s")
#A = read(file, "A")

beta_init = read(file, "beta_init")
mu_init = read(file, "mu_init")
A_init = read(file, "A_init")

close(file)
#@time am = fit(SingleModelAmica,x;maxiter=100, remove_mean = true,m=3,#=M = 1, =#beta=beta_init[:,:,1], mu=mu_init[:,:,1], A=copy(A_init[:,:,1]))
@time am = fit(MultiModelAmica,x;maxiter=100, m=3,M = 2,beta=beta_init, mu=mu_init, A=copy(A_init))

#@time am = fit(SingleModelAmica,x;maxiter=538, m=3)
#@time am = fit(MultiModelAmica,x;maxiter=50,M=2, m=3)
#size(am.A)
#W = pinv(am.models[1].A[:,:]); #previously [:,:,2]

#---
f = Figure()
series(f[1,1],s[:,1:100])
#ax,h = heatmap(f[1,2],A)
#Colorbar(f[1,3],h)

series(f[2,1],x[:,1:100])
#ax,h = heatmap(f[2,2],am.A[:,:,1])
#Colorbar(f[2,3],h)

series(f[3,1],pinv(am.models[1].A[:,:,1])*x[:,1:100])
#series(f[4,1],am.Lt)
#series(f[4,1],am.LL)

#series(f[4,2],(W*x)[:,1:100])

f
#series(f[4,1],pinv(W)'*x)
#series(f[6,1],W'*x)

#----

#_____________________________________________________________
# #use eeg from .mat
# file = matopen("test/eeg_data.mat")
# x = read(file, "x")

# beta_init = read(file, "beta_init")
# mu_init = read(file, "mu_init")
# A_init = read(file, "A_init")

# close(file)

# am = fit(MultiModelAmica,x;maxiter=20,M=1, m=3, beta=beta_init, mu=mu_init, A=A_init)

#plot
# size(am.A)
# W = pinv(am.A[:,:,1]) #previously [:,:,2]

# f = Figure()
# series(f[1,1],d[1:7,1:100])

# ax,h = heatmap(f[1,2],am.A[:,:,1])
# Colorbar(f[1,3],h)

# unmixed_d = am.A[:,:,1]*d
# series(f[3,1],pinv(unmixed_d[1:7,1:100]))
# series(f[4,1],am.Lt)
# series(f[4,2],am.LL)