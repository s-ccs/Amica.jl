function calculate_LL!(myAmica::AbstractAmica, iter) #lines 225 - 231
	LL = calculate_LL(myAmica.Lt,myAmica.M,myAmica.N,myAmica.n)
	myAmica.LL[iter] = LL
	
	return myAmica
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


function loglikelihoodMMGG(loc::AbstractMatrix,scale::AbstractMatrix,shape::AbstractMatrix,mixtureproportions::AbstractMatrix,data::AbstractMatrix)
	return hcat(loglikelihoodMMGG.( eachcol(loc),
                    eachcol(scale),
                    eachcol(shape),
                    eachcol(mixtureproportions),
                    eachrow(data))...)

end
function loglikelihoodMMGG(location::AbstractVector,scale::AbstractVector,shape::AbstractVector,mixtureproportions::AbstractVector,data::AbstractVector)
	MM = MMGG(location,scale,shape,mixtureproportions)
	return loglikelihood.(MM,data)
end
# calculate loglikelihood for each sample in vector x, given a parameterization of a mixture of PGeneralizedGaussians
function MMGG(location::AbstractVector,scale::AbstractVector,shape::AbstractVector,mixtureproportions::AbstractVector)
	# take the vectors of μ,α,ρ and generate a GG from each
    GGvec = PGeneralizedGaussian.(location,scale,shape;check_args=false)
    MM = MixtureModel(GGvec,Vector(mixtureproportions)) # make it a mixture model with prior probabilities π
	return MM
  #  return loglikelihood.(MM,data) # apply the loglikelihood to each sample individually (note the "." infront of .(MM,x))
end


GMM()

function calculate_Lt!(myAmica, h)
	myAmica.ldet[h] =  -log(abs(det(myAmica.A[:,:,h])))
	myAmica.Lt[h,:] .= log(myAmica.proportions[h]) + myAmica.ldet[h]

	Lt_h = myAmica.Lt[h,:]'
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




# function calculate_Lt!(Lt_h, Q, y, n, m, h, learnedParameters::GGParameters)
# 	@error "unlikely that this function is still working ;)"
# 	Lt_h = Lt_h'
# 	for i in 1:n
# 		for j in 1:m
# 			Q[j,:] = log.(learnedParameters.α[j,i,h]) + 0.5*log.(learnedParameters.β[j,i,h]) .+ logpfun(y[i,:,j,h],learnedParameters.ρ[j,i,h])
# 		end
# 		if m > 1
# 			Qmax = ones(m,1).*maximum(Q,dims=1);
# 			Lt_h = Lt_h .+ Qmax[1,:]' .+ log.(sum(exp.(Q - Qmax),dims = 1))
# 		else
# 			Lt_h = Lt_h .+ Q[1,:]
# 		end
# 	end
	
# 	return Lt_h
# end

