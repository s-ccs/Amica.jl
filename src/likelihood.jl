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
function loopiloop!(myAmica::SingleModelAmica{T,ncomps,nmix}, y_rho) where {T,ncomps,nmix}
	N = size(myAmica.source_signals,2)
	n = size(myAmica.source_signals,1)
	Q = Array{T}(undef,nmix,N)
	gg = myAmica.learnedParameters
	for i in 1:n
		calculate_Q!(Q,gg.proportions[:,i],gg.scale[:,i],gg.shape[:,i],@view(y_rho[:,i,:]))
		@debug :Q, Q[1,1],gg.proportions[1,i],gg.scale[1,i],gg.shape[1,i],myAmica.y[1,i,1] #,@view(y_rho[comps,:,:]))
		#Q = calculate_Q(myAmica,i, y_rho) # myAmica.Q
		calculate_u!(myAmica,Q,i) # myAmica.z
		@debug "z",myAmica.z[1,i,1]
		calculate_Lt!(myAmica,Q) # myAmica.Q
	end
	@debug "lt",myAmica.Lt[[1,end]]
end

function loopiloop!(myAmica::MultiModelAmica, y_rho)
	M = size(myAmica.models,1)
	(n,_) = size(myAmica.models[1].source_signals)

	for h in 1:M #run along models
		for i in 1:n #run along components
			Q = calculate_Q(myAmica.models[h], i, y_rho)
			calculate_u!(myAmica.models[h], @view(Q[:, n, :]),i)
			calculate_Lt!(myAmica.models[h], @view(Q[:, n, :]))
		end
	end
end

function calculate_Q(myAmica::SingleModelAmica, i, y_rho)
	(n,N) = size(myAmica.source_signals)
	m = myAmica.m
	Q = zeros(m,N)
	gg = myAmica.learnedParameters
	@views calculate_Q!(Q,gg.proportions[:,i],gg.scale[:,i],gg.shape[:,i],y_rho[:,i,:])
	return Q
end

function calculate_Q!(Q,proportions,scale,shape,y_rho)
	for j in eachindex(proportions)
		Q[j,:] .= log(proportions[j]) + 0.5 * log(scale[j]) .+ logpfun(shape[j], y_rho[j, :])
	end

	return Q
end

#calculates u but saves it into z. MultiModel also uses the SingleModel version
 function calculate_u!(myAmica::SingleModelAmica{T,ncomps,nmix}, Q, i) where {T,ncomps,nmix}
	m = size(myAmica.learnedParameters.scale,1)
	#scratch_Q = similar(@view(Q[1,:]))
	#scratch_Q = similar(Q)
	tmp = Array{T}(undef,size(Q,1),size(Q,2))
	if m > 1
		# i is component/channel
		ixes = 1:m
		for j in ixes # 1:3 m mixtures
			# I think it is very hard to optimize the Q .- Q[j,:] inner term
#			ix = (ixes)[ixes .∈ 2]
			tmp .= @view(Q[:,:]) .- @view(Q[j, :])'
			#tmp .= exp.(tmp)
			optimized_exp!(tmp)

			sum!(@view(myAmica.z[j, i, :])', tmp)
		end
	end
	myAmica.z[:,i,:] .= 1 ./ @view(myAmica.z[:,i,:])
end

#Applies location and scale parameter to source signals (per generalized Gaussian)
@views function calculate_y!(myAmica::SingleModelAmica)
		for j in 1:myAmica.m
			myAmica.y[j, :, :] .= sqrt.(myAmica.learnedParameters.scale[j, :]) .* (myAmica.source_signals .- myAmica.learnedParameters.location[j, :])
		end
end

function calculate_y!(myAmica::MultiModelAmica)
		calculate_y!.(myAmica.models[h])
end

#Calculates Likelihood for each time sample and for each ICA model
function calculate_Lt!(myAmica::SingleModelAmica,Q)
	m = size(myAmica.learnedParameters.scale,1)
	if m > 1
		myAmica.Lt .+= logsumexp(Q;dims=1)[1,:]
	else
		myAmica.Lt[:] = myAmica.Lt .+ Q[1,:] #todo: test
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


