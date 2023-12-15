#Calculates log-likelihood for whole model. todo: make the calculate LLs one function
function calculate_LL!(myAmica::SingleModelAmica)
	(n,N) = size(myAmica.source_signals)
	push!(myAmica.LL,sum(myAmica.Lt) / (n*N))
end

#Calculates LL for whole ICA mixture model. Uses LL for dominant model for each time sample.
function calculate_LL!(myAmica::MultiModelAmica)
	M = size(myAmica.models,1)
	(n,N) = size(myAmica.models[1].source_signals)
	Ltmax = ones(size(myAmica.models[1].Lt,1)) #Ltmax = (M x N)
	Lt_i = zeros(M)
	P = zeros(N)
	for i in 1:N
		for h in 1:M
			Lt_i[h] = myAmica.models[h].Lt[i]
		end
		Ltmax[i] = maximum(Lt_i) #Look for the maximum ith entry among all models
		for h in 1:M
			P[i] = P[i] + exp(myAmica.models[h].Lt[i] - Ltmax[i])
		end
	end 
	push!(myAmica.LL, sum(Ltmax .+ log.(P)) / (n*N))
end

#Update loop for Lt and u (which is saved in z). Todo: Rename
function loopiloop!(myAmica::SingleModelAmica)
	(n,N) = size(myAmica.source_signals)
	m = myAmica.m
	Q = zeros(m,N)
	@debug (:prop,myAmica.learnedParameters.proportions[1,1],:shape,myAmica.learnedParameters.shape[1],:y,myAmica.y[1])
	for i in 1:n
		Q = calculate_Q(myAmica,Q,i) # myAmica.Q
		calculate_u!(myAmica,Q,i) # myAmica.z
		@debug (:z,myAmica.z[1,1:5,1])
		calculate_Lt!(myAmica,Q) # myAmica.Q
	end
	@debug (:Q,Q[1,1])
end

function loopiloop!(myAmica::MultiModelAmica)
	M = size(myAmica.models,1)
	(n,_) = size(myAmica.models[1].source_signals)

	for h in 1:M #run along models
		
		Threads.@threads for i in 1:n #run along components
			Q = calculate_Q(myAmica.models[h], i)
			calculate_u!(myAmica.models[h], @view(Q[:, n, :]),i)
			calculate_Lt!(myAmica.models[h], @view(Q[:, n, :]))
		end
	end
end

function calculate_Q(myAmica::SingleModelAmica, i)
	(n,N) = size(myAmica.source_signals)
	m = myAmica.m
	Q = zeros(m,N)
	
	for j in 1:m
		Q[j,:] .= log(myAmica.learnedParameters.proportions[j,i]) + 0.5 * log(myAmica.learnedParameters.scale[j,i]) .+ logpfun(myAmica.y[i,:,j], myAmica.learnedParameters.shape[j,i])
	end

	return Q
end

#calculates u but saves it into z. MultiModel also uses the SingleModel version
@views function calculate_u!(myAmica::SingleModelAmica, Q, i)
	m = size(myAmica.learnedParameters.scale,1)
	if m > 1
		for j in 1:m
            myAmica.z[i, :, j] .= (1 ./ sum(optimized_exp(Q .- Q[j, :]'), dims=1))[:]
		end
	end
end

#Applies location and scale parameter to source signals (per generalized Gaussian)
function calculate_y!(myAmica::SingleModelAmica)
	for j in 1:myAmica.m
		myAmica.y[:,:,j] .= sqrt.(myAmica.learnedParameters.scale[j,:]) .* (myAmica.source_signals[:,:] .- myAmica.learnedParameters.location[j,:])
	end
end

@views function calculate_y!(myAmica::MultiModelAmica)
	n = size(myAmica.models[1].A,1)
	for h in 1:size(myAmica.models,1)
		#=Threads.@threads=# for i in 1:n
			for j in 1:myAmica.m
				myAmica.models[h].y[i,:,j] = sqrt(myAmica.models[h].learnedParameters.scale[j,i]) * (myAmica.models[h].source_signals[i,:] .- myAmica.models[h].learnedParameters.location[j,i])
			end
		end
	end
end



#Calculates densities for each generalized Gaussian j. Currently used my MultiModelAmica too
function calculate_Q(myAmica::SingleModelAmica, Q, i)
	m = size(myAmica.learnedParameters.scale, 1) #m = number of GGs, can't use myAmica.m in case this gets used my MultiModelAmica

	for j in 1:m
		Q[j,:] = log(myAmica.learnedParameters.proportions[j,i]) + 0.5*log(myAmica.learnedParameters.scale[j,i]) .+ logpfun(myAmica.y[i,:,j],myAmica.learnedParameters.shape[j,i])
	end
	return Q
end


#Calculates Likelihood for each time sample and for each ICA model
function calculate_Lt!(myAmica::SingleModelAmica,Q)
	m = size(myAmica.learnedParameters.scale,1)
	if m > 1
		Qmax = ones(m,1).*maximum(Q,dims=1);
		myAmica.Lt[:] = myAmica.Lt' .+ Qmax[1,:]' .+ logsumexp(Q - Qmax,dims = 1)
	else
		myAmica.Lt[:] = myAmica.Lt .+ Q[1,:]#todo: test
	end
end

#no longer in use, multimodel amica also uses singlemodel version
function calculate_Lt!(myAmica::MultiModelAmica,Q,h)
	m = myAmica.m

	if m > 1
		Qmax = ones(m,1).*maximum(Q,dims=1);
		myAmica.models[h].Lt[:] = myAmica.models[h].Lt[:]' .+ Qmax[1,:]' .+ log.(sum(exp.(Q - Qmax),dims = 1))
	else
		myAmica.models[h].Lt[:] = myAmica.models[h].Lt[:] .+ Q[1,:] #todo: test
	end
end

#Initializes the likelihoods for each time sample with the determinant of the mixing matrix
function initialize_Lt!(myAmica::SingleModelAmica)
	myAmica.Lt .= myAmica.ldet
end

#Initializes the likelihoods of each time sample with the determinant of the mixing matrix and the weights for each ICA model
function initialize_Lt!(myAmica::MultiModelAmica)
	M = size(myAmica.models,1)
	for h in 1:M
		myAmica.models[h].Lt .= log(myAmica.normalized_ica_weights[h]) + myAmica.models[h].ldet
	end
end

# calculate loglikelihood for each sample in vector x, given a parameterization of a mixture of PGeneralizedGaussians (not in use)
function loglikelihoodMMGG(μ::AbstractVector,prop::AbstractVector,shape::AbstractVector,data::AbstractVector,π::AbstractVector)
	# take the vectors of μ,prop,shape and generate a GG from each
    GGvec = PGeneralizedGaussian.(μ,prop,shape)
    MM = MixtureModel(GGvec,Vector(π)) # make it a mixture model with prior probabilities π
    return loglikelihood.(MM,data) # apply the loglikelihood to each sample individually (note the "." infront of .(MM,x))
end


