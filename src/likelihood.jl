

"Calculate difference in loglikelihood between iterations"
function calculate_DLL!(dLL, myAmica::SingleModelAmica, iter)
    if iter > 1
        dLL[iter] = myAmica.LL[iter] - myAmica.LL[iter-1]
    end
end


function calculate_u_and_Lt!(myAmica::MultiModelAmica)
    calculate_u_and_Lt!.(myAmica.models)
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

    # Q = qconst * abs(y)^rho 
    # Q = qconst * abs(y)^(rho - 1) * abs(y)
    @timeit to "Q" myAmica.scratch .= push_dimension(QConst) .- (myAmica.y_rho .* abs.(myAmica.y))

    @timeit to "broadcast" begin
        # Find max for numerical stability (over mixture components, dim 3)
        Qmax = maximum(myAmica.scratch, dims=3)

        # Compute logsumexp: Qmax + log(sum(exp(Q - Qmax)))
        logsumexp_Q = Qmax .+ log.(sum(exp.(myAmica.scratch .- Qmax), dims=3))

        # Compute z = exp(Q - logsumexp) + epsilon (same as Fortran: 1/exp(logsumexp - Q) + eps)
        myAmica.z .= exp.(myAmica.scratch .- logsumexp_Q) .+ T(1e-15)

        # Normalize z so that sum over mixtures = 1
        myAmica.z ./= sum(myAmica.z, dims=3)

        # Accumulate Lt: sum logsumexp over channels (dim 2)
        myAmica.Lt .+= dropdims(sum(logsumexp_Q, dims=(2, 3)), dims=(2, 3))

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
end

function calculate_y!(myAmica::MultiModelAmica)
    for model in myAmica.models
        calculate_y!.(model)
    end
end


