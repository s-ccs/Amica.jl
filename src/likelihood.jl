

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

@kernel function calculate_u_kernel!(
    z::DenseArray{T},
    Lt::DenseArray{T},
    @Const(y_rho::DenseArray{T}),
    @Const(QConst::DenseArray{T})
) where {T<:Real}
    k, i = @index(Global, NTuple)

    N, n, m = size(z)

    # compute Q

    # # Find max for numerical stability
    Qmax = QConst[i, 1] .- y_rho[k, i, 1]
    for j in 2:m
        Q = QConst[i, j] .- y_rho[k, i, j]
        if Q > Qmax
            Qmax = Q
        end
    end

    # Compute sum(exp(Q - Qmax))
    sum_exp = zero(T)
    for j in 1:m
        Q = QConst[i, j] .- y_rho[k, i, j]
        sum_exp += exp(Q - Qmax)
    end


    logsumexp_Q = Qmax + log(sum_exp)

    if m > 1
        # Compute z = exp(Q - logsumexp) + e
        z_sum = zero(T)
        for j in 1:m
            Q = QConst[i, j] .- y_rho[k, i, j]
            z[k, i, j] = exp(Q - logsumexp_Q) + T(1e-15)
            z_sum += z[k, i, j]
        end

        # Normalize z so that sum(z[:,i,k]) = 1s
        for j in 1:m
            z[k, i, j] /= z_sum
        end

    else
        for j in 1:m
            z[k, i, j] /= 1 / z[k, i, j]
        end
    end

    # Accumulate Lt & LL
    Atomix.@atomic Lt[k] += logsumexp_Q
end

"Calculates u (saved in z) and Lt contribution in a single pass to avoid duplicate logsumexp computation"
@views function calculate_u_and_Lt!(myAmica::SingleModelAmica{T}) where {T<:Real}
    N, n = size(myAmica.z)

    # Initialize Lt with base values
    ldet = -logabsdet(myAmica.A |> Array)[1]

    myAmica.Lt .= ldet .+ myAmica.LLdetS

    QConst = .-log(T(2)) .- loggamma.(T(1) .+ T(1) ./ (myAmica.shape |> Array)) .+ log.(myAmica.proportions |> Array) .+ log.(myAmica.scale |> Array) |> typeof(myAmica.source_signals)

    backend = KernelAbstractions.get_backend(myAmica.source_signals)
    kernel! = calculate_u_kernel!(backend)
    kernel!(myAmica.z, myAmica.Lt, myAmica.y_rho, QConst, ndrange=(N, n))

    push!(myAmica.LL, sum(myAmica.Lt) / (N * n))
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


