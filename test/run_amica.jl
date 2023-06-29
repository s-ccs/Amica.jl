using Amica
using CairoMakie
using MAT
using LinearAlgebra

# #Hurra für Hardcoding
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

#______________________________________________________________________________________________________________________

# t = range(0,20*π,length=1000)
# #s = sin.(t * collect(0.5:0.8:pi)')'#rand(10,10000)
# using SignalAnalysis
# s = rand(PinkGaussian(length(t)),4)'
# s = s .* [1,2,3,4]
# #A = rand(size(s,1),size(s,1))
# A = [1 1 0 0; 0 1 1 0; 0 0 1 1; 1 0 1 0]
# x = A*s

# f = Figure()
# series(f[1,1],s)
# series(f[2,1],x)
# heatmap(f[1,2],A)
# heatmap(f[2,2],pinv(A))


# am = fit(MultiModelAmica,x;maxiter=500)
# W = am.A[:,:,1]

# series(f[3,1],pinv(W)*x)
# heatmap(f[3,2],pinv(W))
# series(f[4,1],(W)*x)
# heatmap(f[4,2],(W))

# series(f[5,1],am.Lt)
# series(f[5,2],am.LL)


# #series(f[4,1],pinv(W)'*x)
# #series(f[6,1],W'*x)

# f

#______________________________________________________________________________________________________________________

file = matopen("test/pink_sinus_data.mat")
x = read(file, "x")
s = read(file, "s")
A = read(file, "A")

beta_init = read(file, "beta_init")
mu_init = read(file, "mu_init")
A_init = read(file, "A_init")

close(file)

am = fit(MultiModelAmica,x;maxiter=1000,M=2, m=3, beta=beta_init, mu=mu_init, A=A_init)
size(am.A)
W = pinv(am.A[:,:,1]) #previously [:,:,2]


#---
f = Figure()
series(f[1,1],s[:,1:100])
ax,h = heatmap(f[1,2],A)
Colorbar(f[1,3],h)

series(f[2,1],x[:,1:100])
ax,h = heatmap(f[2,2],am.A[:,:,1])
Colorbar(f[2,3],h)

series(f[3,1],W*x[:,1:100])
series(f[4,1],am.Lt)
series(f[4,2],am.LL)

#series(f[4,2],(W*x)[:,1:100])

f
#series(f[4,1],pinv(W)'*x)
#series(f[6,1],W'*x)

#----