#todo: make the calculate LLs one function
function calculate_LL!(myAmica::SingleModelAmica, iter)
	N = myAmica.N
	n = myAmica.n
	myAmica.LL[iter] = sum(myAmica.Lt) / (n*N)
end

function calculate_LL!(myAmica::MultiModelAmica, iter) #todo: fix whole thing. requires Lt for all models at once
	N = myAmica.N
	n = myAmica.n
	Ltmax = ones(size(myAmica.Lt))
	for i in 1:N
		Ltmax[:,i] .= maximum(myAmica.Lt[:,i])
	end
	P = sum(exp.(myAmica.Lt-Ltmax),dims = 1)'
	myAmica.LL[iter] = sum(Ltmax[1,:] + log.(P)) / (n*N) #todo: fix with tuple
end

#old version (not standalone, was used by calculate_LL!)
function calculate_LL(Lt, N, n) #lines 225 - 231
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
	return  (.-abs.(x).^rho .- log(2) .- loggamma.(1+1/rho))
end

function ffun(x,rho) #taken from amica.m
	return rho .* sign.(x) .* abs.(x) .^(rho.-1)
end

# calculate loglikelihood for each sample in vector x, given a parameterization of a mixture of PGeneralizedGaussians
function loglikelihoodMMGG(μ::AbstractVector,prop::AbstractVector,shape::AbstractVector,data::AbstractVector,π::AbstractVector)
	# take the vectors of μ,prop,shape and generate a GG from each
    GGvec = PGeneralizedGaussian.(μ,prop,shape)
    MM = MixtureModel(GGvec,Vector(π)) # make it a mixture model with prior probabilities π
    return loglikelihood.(MM,data) # apply the loglikelihood to each sample individually (note the "." infront of .(MM,x))
end


