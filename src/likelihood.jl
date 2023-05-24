mutable struct GGParameters
	α::AbstractArray{Float64} #source density mixture proportions
	β::AbstractArray{Float64} #source density inverse scale parameter
	μ::AbstractArray{Float64} #source density location parameter
	ρ::AbstractArray{Float64} #source density shape paramters
end

function calculate_LL(Lt, M, N, n) #lines 225 - 231
	if M > 1
		Ltmax = ones(size(Lt))
		for i in 1:N
			Ltmax[:,i] .= maximum(Lt[:,i])
		end
		P = sum(exp.(Lt-Ltmax),dims = 1)'
		return sum(Ltmax[1,:] + log.(P)) / (n*N)
	else 
		return sum(Lt) / (n*N)
	end
end

function logpfun(x,rho) #taken from amica.m
	return  (-abs.(x).^rho .- log(2) .- loggamma(1+1/rho))
end

function ffun(x,rho) #taken from amica.m
	return rho * sign.(x) .* abs.(x) .^(rho-1)
end

# calculate loglikelihood for each sample in vector x, given a parameterization of a mixture of PGeneralizedGaussians
function loglikelihoodMMGG(μ::AbstractVector,α::AbstractVector,ρ::AbstractVector,x::AbstractVector,π::AbstractVector)
	# take the vectors of μ,α,ρ and generate a GG from each
    GGvec = PGeneralizedGaussian.(μ,α,ρ)
    MM = MixtureModel(GGvec,Vector(π)) # make it a mixture model with prior probabilities π
    return loglikelihood.(MM,x) # apply the loglikelihood to each sample individually (note the "." infront of .(MM,x))
end

function calculate_Lt!(Lt_h, Q, y, n, m, h, learnedParameters::GGParameters)
	Lt_h = Lt_h'
	for i in 1:n
		for j in 1:m
			Q[j,:] = log.(learnedParameters.α[j,i,h]) + 0.5*log.(learnedParameters.β[j,i,h]) .+ logpfun(y[i,:,j,h],learnedParameters.ρ[j,i,h])
		end
		if m > 1
			Qmax = ones(m,1).*maximum(Q,dims=1);
			Lt_h = Lt_h .+ Qmax[1,:]' .+ log.(sum(exp.(Q - Qmax),dims = 1))
		else
			Lt_h = Lt_h .+ Q[1,:]
		end
	end
	
	return Lt_h
end

