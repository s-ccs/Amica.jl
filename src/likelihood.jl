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
    @timeit to "calculate_u" calculate_u!(myAmica.z, myAmica.Q)
    @timeit to "calculate_Lt" calculate_Lt!(myAmica)
    @timeit to "calculate_LL" calculate_LL!(myAmica)
end

function loopiloop!(myAmica::MultiModelAmica)
    M = size(myAmica.models, 1)
    (n, _) = size(myAmica.models[1].source_signals)

    for h in 1:M #run along models
        model = models[h].Q = calculate_Q(myAmica.models[h], i, y_rho)
        gg = model.learnedParameters

        @timeit to "calculate_Q" calculate_Q!(model.Q, gg.proportions, gg.scale, gg.shape, model.y_rho)
        @timeit to "calculate_u" calculate_u!(model.z, model.Q)
        @timeit to "calculate_Lt" calculate_Lt!(model)

    end
    @timeit to "calculate_LL" calculate_LL!(myAmica)
end

function calculate_Q!(Q::AbstractArray{T,3}, proportions::AbstractArray{T,2}, scale::AbstractArray{T,2}, shape::AbstractArray{T,2}, y_rho::AbstractArray{T,3}) where {T<:Real}
    Q .= y_rho
    logpfun!(Q, y_rho, shape)
    # idk why but this is faster when stored in a variable?!
    add = (log.(proportions) .+ log.(scale))
    Q .+= add
end

#calculates u but saves it into z. MultiModel also uses the SingleModel version
@views function calculate_u!(z::AbstractArray{T,3}, Q::AbstractArray{T,3}) where {T<:Real}
    (m, n, N) = size(z)

    if (m <= 1)
        z .= 1 ./ z
        return
    end

    # Compute logsumexp for each (i, k) pair and store in temporary variable
    # This corresponds to tmpvec = Pmax + log(sum(exp(z0 - Pmax))) in Fortran
    for k = 1:N, i = 1:n
        # Find max for numerical stability
        Qmax = Q[1, i, k]
        for j = 2:m
            @inbounds Qmax = max(Qmax, Q[j, i, k])
        end

        # Compute sum(exp(Q - Qmax))
        sum_exp = zero(T)
        for j = 1:m
            @inbounds sum_exp += exp(Q[j, i, k] - Qmax)
        end

        # TODO use logsumexp
        logsumexp_Q = Qmax + log(sum_exp)

        # Compute z[j,i,k] = 1 / exp(logsumexp_Q - Q[j,i,k]) + epsilon
        for j = 1:m
            @inbounds z[j, i, k] = one(T) / exp(logsumexp_Q - Q[j, i, k]) + T(1e-15)
        end

        # Normalize z so that sum(z[:,i,k]) = 1
        z_sum = sum(z[:, i, k])
        for j = 1:m
            @inbounds z[j, i, k] /= z_sum
        end
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

"Calculates the likelihoods for each time sample"
function calculate_Lt!(myAmica::SingleModelAmica)
    myAmica.Lt .= myAmica.ldet
    myAmica.Lt .+= myAmica.LLdetS

    (m, n, N) = size(myAmica.Q)

    if m > 1
        # Manual logsumexp computation without allocations
        for k in 1:N
            log_sum = zero(eltype(myAmica.Q))
            for i in 1:n
                # Find max for numerical stability
                Qmax = myAmica.Q[1, i, k]
                for j in 2:m
                    @inbounds Qmax = max(Qmax, myAmica.Q[j, i, k])
                end

                # Compute logsumexp
                sum_exp = zero(eltype(myAmica.Q))
                for j in 1:m
                    @inbounds sum_exp += exp(myAmica.Q[j, i, k] - Qmax)
                end
                log_sum += Qmax + log(sum_exp)
            end
            @inbounds myAmica.Lt[k] += log_sum
        end
    else
        for k in 1:N
            sum_val = zero(eltype(myAmica.Q))
            for i in 1:n
                @inbounds sum_val += myAmica.Q[1, i, k]
            end
            @inbounds myAmica.Lt[k] += sum_val
        end
    end
end

# calculate loglikelihood for each sample in vector x, given a parameterization of a mixture of PGeneralizedGaussians (not in use)
function loglikelihoodMMGG(μ::AbstractVector, prop::AbstractVector, shape::AbstractVector, data::AbstractVector, π::AbstractVector)
    # take the vectors of μ,prop,shape and generate a GG from each
    GGvec = PGeneralizedGaussian.(μ, prop, shape)
    MM = MixtureModel(GGvec, Vector(π)) # make it a mixture model with prior probabilities π
    return loglikelihood.(MM, data) # apply the loglikelihood to each sample individually (note the "." infront of .(MM,x))
end


