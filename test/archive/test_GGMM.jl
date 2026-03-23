#---
using CairoMakie
using SignalAnalysis
using Amica
#t = range(0,20*π,length=1000)
s = rand(PinkGaussian(length(t)), 4)'
#s[2,:] = sin.(t)
#s[3,:] = sin.(2 .* t)
#s[4,:] = sin.(10 .* t)
#s = s .* [1,2,3,4]
function fill_gmm(n)
       gm = rand(GMM, 3, 2)
       return Vector(rand(MixtureModel(gm), n)[1, :])
end
#A = rand(size(s,1),size(s,1))
s[2, :] = fill_gmm(1000)
s[3, :] = fill_gmm(1000)
s[4, :] = fill_gmm(1000)

A = [1 1 0 0; 0 1 1 0; 0 0 1 1; 1 0 1 0]

x = A * s

#A = [1 1 0 1; 1 1 0 0; 1 0 1 1; 0 0 0 1]
#x = hcat(x,A*s) 

am = fit(MultiModelAmica, x; maxiter=2000, M=1)

#---
scale = am.β #scale => GG.alpha
mixtureproportions = am.α #GG.prior
shape = am.ρ #shape => GG.rho
location = am.μ #location
data = am.source_signals

# calculate loglikelihood for each sample in vector x, given a parameterization of a mixture of PGeneralizedGaussians (not in use)
function loglikelihoodMMGG(μ::AbstractVector, prop::AbstractVector, shape::AbstractVector, data::AbstractVector, π::AbstractVector)
       # take the vectors of μ,prop,shape and generate a GG from each
       GGvec = PGeneralizedGaussian.(μ, prop, shape)
       MM = MixtureModel(GGvec, Vector(π)) # make it a mixture model with prior probabilities π
       return loglikelihood.(MM, data) # apply the loglikelihood to each sample individually (note the "." infront of .(MM,x))
end


l = loglikelihoodMMGG(location[:, :, 1],
       scale[:, :, 1],
       shape[:, :, 1],
       mixtureproportions[:, :, 1],
       data[:, :, 1])




MM = Amica.MMGG(location[:, 1, 1],
       scale[:, 1, 1],
       shape[:, 1, 1],
       mixtureproportions[:, 1, 1],
       data[1, :, 1])

#---
f = Figure()
ax = f[1, 1] = Axis(f)
for n = 1:4
       MM = Amica.MMGG(location[:, n, 1],
              (scale[:, n, 1]),
              #[1,1,1.],
              shape[:, n, 1],
              mixtureproportions[:, n, 1])

       x_t = -5:0.1:5

       lines!(x_t, pdf.(Ref(MM), x_t))
end
n = 2
hist!(data[n, :, 1], bins=x_t; normalization=:pdf, color=RGBAf(1, 0, 0, 0.2))
current_figure()
#---
#=
j = 1
i = 1
h = 1

Q = zeros(3,1000)
for j in 1:3

Q[j,:] = log.(am.α[j,i,h]) + 0.5*log.(am.β[j,i,h]) .+ Amica.logpfun(am.y[i,:,j,h],am.ρ[j,i,h])
end
Qmax = ones(3,1).*maximum(Q,dims=1);
Qmax[1,:]' .+ log.(sum(exp.(am.Q - Qmax),dims = 1))
=#