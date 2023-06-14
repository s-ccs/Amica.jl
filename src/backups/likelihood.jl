function calculate_LL(myAmica, iter) #lines 225 - 231
	M = myAmica.M
	N = myAmica.n
	n = myAmica.n
	if M > 1
		Ltmax = ones(size(myAmica.Lt))
		for i in 1:N
			Ltmax[:,i] .= maximum(myAmica.Lt[:,i])
		end
		P = sum(exp.(myAmica.Lt-Ltmax),dims = 1)'
		myAmica.LL[iter] = sum(Ltmax[1,:] + log.(P)) / (n*N)
	else 
		myAmica.LL[iter] = sum(myAmica.Lt) / (n*N)
	end
	return myAmica
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

function calculate_Lt!(myAmica, h)

	myAmica.ldet[h] =  -log(abs(det(myAmica.A[:,:,h])))
	myAmica.Lt[h,:] .= log(myAmica.proportions[h]) + myAmica.ldet[h]

	Lt_h = myAmica.Lt[h]'
	n = myAmica.n
	m = myAmica.m
	for i in 1:n
		for j in 1:m
			myAmica.Q[j,:] = log.(myAmica.learnedParameters.α[j,i,h]) + 0.5*log.(myAmica.learnedParameters.β[j,i,h]) .+ logpfun(myAmica.y[i,:,j,h],myAmica.learnedParameters.ρ[j,i,h])
		end
		if m > 1
			Qmax = ones(m,1).*maximum(myAmica.Q,dims=1);
			Lt_h = Lt_h .+ Qmax[1,:]' .+ log.(sum(exp.(myAmica.Q - Qmax),dims = 1))
		else
			Lt_h = Lt_h .+ myAmica.Q[1,:]
		end
	end
	
	myAmica.Lt[h,:] = Lt_h
	return myAmica
end

