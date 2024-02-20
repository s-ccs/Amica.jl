#Calculates log-likelihood for whole model. todo: make the calculate LLs one function
function calculate_LL!(myAmica::SingleModelAmica)
    (n, N) = size(myAmica.source_signals)
    push!(myAmica.LL, sum(myAmica.Lt) / (n * N))
end

#Calculates LL for whole ICA mixture model. Uses LL for dominant model for each time sample.
function calculate_LL!(myAmica::MultiModelAmica)
    M = size(myAmica.models, 1)
    (n, N) = size(myAmica.models[1].source_signals)
    Ltmax = ones(size(myAmica.models[1].Lt, 1)) #Ltmax = (M x N)
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
    push!(myAmica.LL, sum(Ltmax .+ log.(P)) / (n * N))
end

#Update loop for Lt and u (which is saved in z). Todo: Rename
function loopiloop!(myAmica::SingleModelAmica{T}) where {T}
    gg = myAmica.learnedParameters
    calculate_Q!(myAmica.Q, gg.proportions, gg.scale, gg.shape, myAmica.y_rho)
    calculate_u!(myAmica.z, myAmica.Q, myAmica.u_intermed)
    calculate_Lt!(myAmica.Lt, myAmica.Q)

end

function loopiloop!(myAmica::MultiModelAmica)
    M = size(myAmica.models, 1)
    (n, _) = size(myAmica.models[1].source_signals)

    for h in 1:M #run along models
        for i in 1:n #run along components
            Q = calculate_Q(myAmica.models[h], i, y_rho)
            calculate_u!(myAmica.models[h], @view(Q[:, n, :]), i)
            calculate_Lt!(myAmica.models[h], @view(Q[:, n, :]))
        end
    end
end

function calculate_Q!(Q::AbstractArray{T,3}, proportions::AbstractArray{T,2}, scale::AbstractArray{T,2}, shape::AbstractArray{T,2}, y_rho::AbstractArray{T,3}) where {T<:Real}
    Q .= y_rho
    logpfun!(Q, y_rho, shape)
    # idk why but this is faster when stored in a variable?!
    add = (log.(proportions) .+ 0.5 .* log.(scale))
    Q .+= add
end

#calculates u but saves it into z. MultiModel also uses the SingleModel version
@views function calculate_u!(z::AbstractArray{T,3}, Q::AbstractArray{T,3}, u_intermed::AbstractArray{T,4}) where {T<:Real}
    (m, n, N) = size(z)

    if (m <= 1)
        z .= 1 ./ z
        return
    end

    for k = 1:N, i = 1:n, j = 1:m, j1 = 1:m
        @inbounds u_intermed[j, i, k, j1] = Q[j1, i, k] - Q[j, i, k]
    end

    optimized_exp!(u_intermed)

    sum!(z, u_intermed)

    z .= 1 ./ z
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
function calculate_Lt!(Lt::Array{T,1}, Q::Array{T,3}) where {T<:Real}
    (m, _, _) = size(Q)

    if m > 1
        Lt .+= sum(logsumexp(Q; dims=1), dims=2)[1, 1, :]
    else
        Lt[:] = Lt .+ Q[1, :] #todo: test
    end
end

#Initializes the likelihoods for each time sample with the determinant of the mixing matrix
function initialize_Lt!(myAmica::SingleModelAmica)
    myAmica.Lt .= myAmica.ldet
end

#Initializes the likelihoods of each time sample with the determinant of the mixing matrix and the weights for each ICA model
function initialize_Lt!(myAmica::MultiModelAmica)
    M = size(myAmica.models, 1)
    for h in 1:M
        myAmica.models[h].Lt .= log(myAmica.normalized_ica_weights[h]) + myAmica.models[h].ldet
    end
end

# calculate loglikelihood for each sample in vector x, given a parameterization of a mixture of PGeneralizedGaussians (not in use)
function loglikelihoodMMGG(μ::AbstractVector, prop::AbstractVector, shape::AbstractVector, data::AbstractVector, π::AbstractVector)
    # take the vectors of μ,prop,shape and generate a GG from each
    GGvec = PGeneralizedGaussian.(μ, prop, shape)
    MM = MixtureModel(GGvec, Vector(π)) # make it a mixture model with prior probabilities π
    return loglikelihood.(MM, data) # apply the loglikelihood to each sample individually (note the "." infront of .(MM,x))
end


