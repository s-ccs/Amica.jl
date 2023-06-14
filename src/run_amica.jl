using Amica

x = [1 4; 4 1]*Float64.([1.0 2 3; 4 5 6])
M = 2 #number of mixture models
m = 3 #number of source density mixtures


A = zeros(size(x,1),size(x,1),M)
A[:,:,1] = [1.0 0.003; -0.05 1.0]
A[:,:,2] = [2.0 0.003; -0.05 1.0]

beta = ones(m, size(x,1), M)
beta[:,:,1] = [1.1 0.9; 1.0 0.9; 0.9 0.8]
beta[:,:,2] = [1.2 0.9; 1.1 0.8; 0.9 0.7]

mu = zeros(m, size(x,1), M)
mu[:,:,1] = [0.1 0.9; -0.01 0.0; 0.0 -0.02] #todo: wieder rnd mu einfürgen
mu[:,:,2] = [0.2 1; -0.01 0.0; 0.0 -0.03]

amica = MultiModelAmica(Amica.removeMean(x);M=2,A=A,mu=mu,beta=beta)
Amica.amica!(amica,x;maxiter=5,mindll = 1e-8,iterwin = 1)


fit(MultiModelAmica,x;remove_mean = true)
fit!(amica,x)
#z, A, Lt, LL = amica(x, M, m, maxiter, update_rho, mindll, iterwin, do_newton, remove_mean)



x = rand(10,10000)

fit(MultiModelAmica,x;remove_mean = true)