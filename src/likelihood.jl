"Calculates log-likelihood for whole model. todo: make the calculate LLs one function"
function calculate_LL!(myAmica::SingleModelAmica)
    (n, N) = size(myAmica.source_signals)
    push!(myAmica.LL, sum(myAmica.Lt) / (n * N))
end

"Calculate difference in loglikelihood between iterations"
function calculate_DLL!(dLL, myAmica::SingleModelAmica, iter)
    if iter > 1
        dLL[iter] = myAmica.LL[iter] - myAmica.LL[iter-1]
    end
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
    @timeit to "calculate_Q" calculate_Q!(myAmica.Q, gg.proportions, gg.scale, gg.shape, myAmica.y_rho)
    @timeit to "calculate_u_and_Lt" calculate_u_and_Lt!(myAmica)
    @timeit to "calculate_LL" calculate_LL!(myAmica)
end

function loopiloop!(myAmica::MultiModelAmica)
    M = size(myAmica.models, 1)
    (n, _) = size(myAmica.models[1].source_signals)

    for h in 1:M #run along models
        model = myAmica.models[h]
        gg = model.learnedParameters

        @timeit to "calculate_Q" calculate_Q!(model.Q, gg.proportions, gg.scale, gg.shape, model.y_rho)
        @timeit to "calculate_u_and_Lt" calculate_u_and_Lt!(model)

    end
    @timeit to "calculate_LL" calculate_LL!(myAmica)
end

function calculate_Q!(Q::AbstractArray{T,3}, proportions::AbstractArray{T,2}, scale::AbstractArray{T,2}, shape::AbstractArray{T,2}, y_rho::AbstractArray{T,3}) where {T<:Real}
    # the 'identity' brings a significant speedup
    Q .= identity(-log(2) .- loggamma.(1 .+ 1 ./ shape) .+ log.(proportions) .+ log.(scale)) .- y_rho
end


"Calculates u (saved in z) and Lt contribution in a single pass to avoid duplicate logsumexp computation"
@views function calculate_u_and_Lt!(myAmica::SingleModelAmica{T}) where {T<:Real}
    z = myAmica.z
    Q = myAmica.Q
    Lt = myAmica.Lt
    ldet = myAmica.ldet
    LLdetS = myAmica.LLdetS
    (m, n, N) = size(z)

    # Initialize Lt with base values
    Lt .= ldet .+ LLdetS

    if m <= 1
        z .= 1 ./ z
        # Add Q contribution to Lt for m == 1 case
        for k in 1:N
            sum_val = zero(T)
            for i in 1:n
                sum_val += Q[1, i, k]
            end
            Lt[k] += sum_val
        end
        return
    end

    @inbounds for k in 1:N, i in 1:n
        # Find max for numerical stability
        Qmax = maximum(Q[:, i, k])

        # Compute sum(exp(Q - Qmax))
        sum_exp = zero(T)
        for j in 1:m
            sum_exp += exp(Q[j, i, k] - Qmax)
        end

        logsumexp_Q = Qmax + log(sum_exp)

        # Compute z = exp(Q - logsumexp) + e
        @. z[:, i, k] = exp(Q[:, i, k] - logsumexp_Q) + T(1e-15)

        # Normalize z so that sum(z[:,i,k]) = 1
        z_sum = sum(z[:, i, k])
        z[:, i, k] ./= z_sum

        Lt[k] += logsumexp_Q  # Accumulate for Lt
    end
end


"Applies location and scale parameter to source signals (per generalized Gaussian)"
@views function calculate_y!(myAmica::SingleModelAmica)
    for j in 1:myAmica.m
        myAmica.y[j, :, :] .= myAmica.learnedParameters.scale[j, :] .* (myAmica.source_signals .- myAmica.learnedParameters.location[j, :])
    end
end

function calculate_y!(myAmica::MultiModelAmica)
    calculate_y!.(myAmica.models[h])
end


# calculate loglikelihood for each sample in vector x, given a parameterization of a mixture of PGeneralizedGaussians (not in use)
function loglikelihoodMMGG(μ::AbstractVector, prop::AbstractVector, shape::AbstractVector, data::AbstractVector, π::AbstractVector)
    # take the vectors of μ,prop,shape and generate a GG from each
    GGvec = PGeneralizedGaussian.(μ, prop, shape)
    MM = MixtureModel(GGvec, Vector(π)) # make it a mixture model with prior probabilities π
    return loglikelihood.(MM, data) # apply the loglikelihood to each sample individually (note the "." infront of .(MM,x))
end


