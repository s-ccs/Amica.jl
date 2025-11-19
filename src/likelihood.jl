

"Calculate difference in loglikelihood between iterations"
function calculate_DLL!(dLL, myAmica::SingleModelAmica, iter)
    if iter > 1
        dLL[iter] = myAmica.LL[iter] - myAmica.LL[iter-1]
    end
end


function calculate_u_and_Lt!(myAmica::MultiModelAmica)
    M = size(myAmica.models, 1)
    for h in 1:M #run along models
        @timeit to "calculate_u_and_Lt" calculate_u_and_Lt!(myAmica.models[h])
    end
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

    (N, n, m) = size(z)

    # Initialize Lt with base values
    ldet = -logabsdet(myAmica.A)[1]

    Lt .= ldet .+ LLdetS

    Q = zeros(T, m)

    LL = zero(T)


    QConst = @. -log(T(2)) - loggamma(T(1) + T(1) / shape) + log(proportions) + log(scale)

    @inbounds for i in 1:n, k in 1:N
        # compute Q
        for j in 1:m
            Q[j] = QConst[i, j] - y_rho[k, i, j]
        end

        if m > 1
            # Find max for numerical stability
            Qmax = maximum(Q)

            # Compute sum(exp(Q - Qmax))
            sum_exp = sum(x::T -> exp(x - Qmax), Q)

            logsumexp_Q = Qmax + log(sum_exp)

            # Compute z = exp(Q - logsumexp) + e
            @. z[k, i, :] = exp(Q[:] - logsumexp_Q) + T(1e-15)

            # Normalize z so that sum(z[:,i,k]) = 1
            z_sum = sum(z[k, i, :])
            z[k, i, :] ./= z_sum

        else
            z .= 1 ./ z
        end

        # Accumulate Lt & LL
        Lt[k] += logsumexp_Q
        LL += logsumexp_Q
    end

    push!(myAmica.LL, (ldet + LLdetS) / n + LL / (n * N))
end

"add a unit dimension in front to be able to e.g. broadcast a (1000, 12, 3) with a (12, 3) array, transforms a from (12, 3) to (1, 12, 3)"
function push_dimension(a::AbstractArray)
    reshape(a, 1, size(a)...)
end

"Applies location and scale parameter to source signals (per generalized Gaussian)"
@views function calculate_y!(myAmica::SingleModelAmica)
    (_, _, m) = size(myAmica.z)

    for j in 1:m
        myAmica.y[:, :, j] .= myAmica.scale[:, j]' .* (myAmica.source_signals .- myAmica.location[:, j]')
    end
end

function calculate_y!(myAmica::MultiModelAmica)
    for model in myAmica.models
        calculate_y!.(model)
    end
end


