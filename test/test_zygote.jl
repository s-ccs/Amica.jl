using Amica
using LinearAlgebra
t = range(0,20*π,length=1000)
s =rand(PinkGaussian(length(t)),4)'
s[2,:] = sin.(t)
s[3,:] = sin.(2 .* t)
s[4,:] = sin.(10 .* t)
s = s .* [1,2,3,4]
#A = rand(size(s,1),size(s,1))
A_org = [1 1 0 0; 0 1 1 0; 0 0 1 1; 1 0 1 0]

x = A_org*s


am = MultiModelAmica(x)

scale = am.learnedParameters.β[:,:,1] #scale => GG.alpha
mixtureproportions = am.learnedParameters.α[:,:,1] #GG.prior
shape = am.learnedParameters.ρ[:,:,1] #shape => GG.rho
location = am.learnedParameters.μ[:,:,1] #location
A = am.A[:,:,1]


using Optimization
myFun(pa,tmp) = myFun(pa)

function myFun(pa)
    sources = inv(pa.A)*x
    L1 = log(abs(det(pa.A)))
    mp = pa.mixtureproportions ./ sum(pa.mixtureproportions,dims=1)
    #@show mp
    L2 = Amica.loglikelihoodMMGG( pa.location,
                    pa.scale,
                    pa.shape,
                    mp,
                    sources)
    return -(L1+sum(L2))
end

using OptimizationOptimJL
using ComponentArrays

para =ComponentArray(;A,location,scale,shape,mixtureproportions)
lb = similar(para)
lb .= -Inf
ub =  .- deepcopy(lb)

lb.scale .= 0.0001
lb.shape .= 0.0001
lb.mixtureproportions .= 0
ub.mixtureproportions .= 1

using ModelingToolkit
df = OptimizationProblem(OptimizationFunction(myFun, Optimization.AutoZygote()), para,[];lb = lb,ub = ub)
sol = solve(df,LBFGS(),maxtime =180)

