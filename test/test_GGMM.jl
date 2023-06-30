#---

using CairoMakie
using SignalAnalysis
using Amica
t = range(0,20*π,length=1000)
s =rand(PinkGaussian(length(t)),4)'
s[2,:] = sin.(t)
s[3,:] = sin.(2 .* t)
s[4,:] = sin.(10 .* t)
s = s .* [1,2,3,4]
#A = rand(size(s,1),size(s,1))
A = [1 1 0 0; 0 1 1 0; 0 0 1 1; 1 0 1 0]

x = A*s

#A = [1 1 0 1; 1 1 0 0; 1 0 1 1; 0 0 0 1]
#x = hcat(x,A*s) 

am = fit(MultiModelAmica,x;maxiter=2000,M=1)

#---
scale = am.learnedParameters.β #scale => GG.alpha
mixtureproportions = am.learnedParameters.α #GG.prior
shape = am.learnedParameters.ρ #shape => GG.rho
location = am.learnedParameters.μ #location
data = am.source_signals

l = Amica.loglikelihoodMMGG( location[:,:,1],
                    scale[:,:,1],
                    shape[:,:,1],
                    mixtureproportions[:,:,1],
                    data[:,:,1])




MM = Amica.MMGG( location[:,1,1],
scale[:,1,1],
shape[:,1,1],
mixtureproportions[:,1,1],
data[1,:,1])

#---

n = 1
MM = Amica.MMGG( location[:,n,1],
       scale[:,n,1],
       shape[:,n,1],
       mixtureproportions[:,n,1],
       data[n,:,1])

#=
MM = Amica.MMGG( location[:,1,1],
[0.2,1,1.],
[3,2,2],
[0.1,0.45,0.45],#]#mixtureproportions[:,1,1],
data[1,:,1])
=#


#gm = GMM(3,1)
#gm.μ .= [-1,0,1]
#lGM = em!(gm,data[1,:,1:1])

#MM = MixtureModel(gm)
x_t = -5:0.1:5

lines(x_t,pdf.(Ref(MM),x_t),color=:red)
hist!(data[n,:,1],bins=x_t;normalization=:pdf)
current_figure()
#---
#=
j = 1
i = 1
h = 1

Q = zeros(3,1000)
for j in 1:3

Q[j,:] = log.(am.learnedParameters.α[j,i,h]) + 0.5*log.(am.learnedParameters.β[j,i,h]) .+ Amica.logpfun(am.y[i,:,j,h],am.learnedParameters.ρ[j,i,h])
end
Qmax = ones(3,1).*maximum(Q,dims=1);
Qmax[1,:]' .+ log.(sum(exp.(am.Q - Qmax),dims = 1))
=#