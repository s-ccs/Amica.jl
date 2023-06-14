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


t = range(0,20*π,length=1000)
#s = sin.(t * collect(0.5:0.8:pi)')'#rand(10,10000)
using SignalAnalysis
s =rand(PinkGaussian(length(t)),4)'
s = s .* [1,2,3,4]
#A = rand(size(s,1),size(s,1))
A = [1 1 0 0; 0 1 1 0; 0 0 1 1; 1 0 1 0]
x = A*s

f = Figure()
series(f[1,1],s)
series(f[2,1],x)
heatmap(f[1,2],A)
heatmap(f[2,2],inv(A))
f

am = fit(MultiModelAmica,x;maxiter=500)
W = am.A[:,:,1]

series(f[3,1],inv(W)*x)
heatmap(f[3,2],inv(W))
series(f[4,1],(W)*x)
heatmap(f[4,2],(W))
f
#series(f[4,1],inv(W)'*x)
#series(f[6,1],W'*x)
f