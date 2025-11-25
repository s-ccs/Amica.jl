

"Calculate difference in loglikelihood between iterations"
function calculate_DLL!(dLL, myAmica::SingleModelAmica, iter)
    if iter > 1
        dLL[iter] = myAmica.LL[iter] - myAmica.LL[iter-1]
    end
end


function calculate_u_and_Lt!(myAmica::MultiModelAmica)
    calculate_u_and_Lt!.(myAmica.models)
end

@kernel function calculate_u_kernel!(
    z::DenseArray{T},
    Lt::DenseArray{T},
    @Const(y::DenseArray{T}),
    @Const(shape::DenseArray{T}),
    @Const(QConst::DenseArray{T}),
    ::Val{M}
) where {T<:Real,M}
    k, i = @index(Global, NTuple)

    m = M  # Use compile-time constant

    # compute Q

    Q = @private eltype(z) (M,)

    for j in 1:m
        # compute y^rho
        y_rho_val = exp((shape[i, j]) * log(abs(y[k, i, j])))
        Q[j] = QConst[i, j] - y_rho_val
    end

    # # Find max for numerical stability
    Qmax = Q[1]
    for j in 2:m
        Qmax = max(Qmax, Q[j])
    end

    # Compute sum(exp(Q - Qmax))
    sum_exp = zero(T)
    for j in 1:m
        sum_exp += exp(Q[j] - Qmax)
    end

    logsumexp_Q = Qmax + log(sum_exp)

    if m > 1
        # Compute z = exp(Q - logsumexp) + e
        z_sum = zero(T)
        for j in 1:m
            z[k, i, j] = exp(Q[j] - logsumexp_Q) + T(1e-15)
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
    N, n, m = size(myAmica.y)

    # Initialize Lt with base values
    @timeit to "ldet" begin
        ldet = -logabsdet(myAmica.A |> Array)[1]
        myAmica.Lt .= ldet .+ myAmica.LLdetS
    end

    @timeit to "qconst" QConst = .-log(T(2)) .- (loggamma.(T(1) .+ T(1) ./ myAmica.shape)) .+ log.(myAmica.proportions) .+ log.(myAmica.scale)

    @timeit to "kernel" begin
        backend = KernelAbstractions.get_backend(myAmica.source_signals)
        kernel! = calculate_u_kernel!(backend)
        kernel!(myAmica.z, myAmica.Lt, myAmica.y, myAmica.shape, QConst, Val(m), ndrange=(N, n))
    end

    if NAN_CHECK_ACTIVE && any(isnan, myAmica.Lt)
        @warn "NaN in myAmica.Lt"
    end
    if NAN_CHECK_ACTIVE && any(isnan, myAmica.z)
        @warn "NaN in myAmica.z"
    end

    @timeit to "sum" push!(myAmica.LL, sum(myAmica.Lt) / (N * n))
end

"add a unit dimension in front to be able to e.g. broadcast a (1000, 12, 3) with a (12, 3) array, transforms a from (12, 3) to (1, 12, 3)"
function push_dimension(a::AbstractArray)
    reshape(a, 1, size(a)...)
end



"Applies location and scale parameter to source signals (per generalized Gaussian)"
@views function calculate_y!(myAmica::SingleModelAmica{T}) where T<:Real
    (_, _, m) = size(myAmica.z)

    for j in 1:m
        myAmica.y[:, :, j] .= myAmica.scale[:, j]' .* (myAmica.source_signals .- myAmica.location[:, j]')
    end

    if NAN_CHECK_ACTIVE && any(isnan, myAmica.y)
        @warn "NaN in myAmica.y"
    end

end

function calculate_y!(myAmica::MultiModelAmica)
    for model in myAmica.models
        calculate_y!.(model)
    end
end


