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
function loopiloop!(myAmica::SingleModelAmica)
    @timeit to "calculate_u_and_Lt" calculate_u_and_Lt!(myAmica)
    @timeit to "calculate_LL" calculate_LL!(myAmica)
end

function loopiloop!(myAmica::MultiModelAmica)
    M = size(myAmica.models, 1)
    (n, _) = size(myAmica.models[1].source_signals)

    for h in 1:M #run along models
        model = myAmica.models[h]

        @timeit to "calculate_u_and_Lt" calculate_u_and_Lt!(model)

    end
    @timeit to "calculate_LL" calculate_LL!(myAmica)
end

"Calculates u (saved in z) and Lt contribution in a single pass to avoid duplicate logsumexp computation"
@views function calculate_u_and_Lt!(myAmica::SingleModelAmica{T}) where {T<:Real}
    z = myAmica.z
    Lt = myAmica.Lt
    LLdetS = myAmica.LLdetS
    proportions = myAmica.proportions
    scale = myAmica.scale
    shape = myAmica.shape
    y_rho = myAmica.y_rho

    (m, n, N) = size(z)

    # Initialize Lt with base values
    ldet = -logabsdet(myAmica.A)[1]
    @info size(ldet)

    Lt .= ldet .+ LLdetS

    Q = zeros(T, m)

    @inbounds for k in 1:N, i in 1:n
        # compute Q
        for j in 1:m
            Q[j] = -log(T(2)) - loggamma(T(1) + T(1) / shape[j, i]) + log(proportions[j, i]) + log(scale[j, i]) - y_rho[j, i, k]
        end

        if m > 1
            # Find max for numerical stability
            Qmax = maximum(Q)

            # Compute sum(exp(Q - Qmax))
            sum_exp = zero(T)
            for j in 1:m
                sum_exp += exp(Q[j] - Qmax)
            end

            logsumexp_Q = Qmax + log(sum_exp)

            # Compute z = exp(Q - logsumexp) + e
            @. z[:, i, k] = exp(Q[:] - logsumexp_Q) + T(1e-15)

            # Normalize z so that sum(z[:,i,k]) = 1
            z_sum = sum(z[:, i, k])
            z[:, i, k] ./= z_sum

        else
            z .= 1 ./ z
        end

        Lt[k] += logsumexp_Q  # Accumulate for Lt
    end
end


"Applies location and scale parameter to source signals (per generalized Gaussian)"
@views function calculate_y!(myAmica::SingleModelAmica)
    (m, _, _) = size(myAmica.z)

    for j in 1:m
        myAmica.y[j, :, :] .= myAmica.scale[j, :] .* (myAmica.source_signals .- myAmica.location[j, :])
    end
end

function calculate_y!(myAmica::MultiModelAmica)
    for model in myAmica.models
        calculate_y!.(model)
    end
end


