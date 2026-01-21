"Normalizes source density location parameter (mu), scale parameter (beta) and model centers"
function reparameterize!(myAmica::SingleModelAmica{T}) where T<:Real
    # Calculate norm of column k: Anrmk = sqrt(sum(A(:,k)*A(:,k)))
    tau = sqrt.(sum(myAmica.A .^ 2, dims=1))[1, :]

    mask = tau .> zero(T)
    myAmica.A .= ifelse.(push_dimension(mask), myAmica.A ./ tau', myAmica.A)
    myAmica.location .= ifelse.(mask, myAmica.location .* tau, myAmica.location)
    myAmica.scale .= ifelse.(mask, myAmica.scale ./ tau, myAmica.scale)
end

"Reparameterizes the parameters for the active models"
function reparameterize!(myAmica::MultiModelAmica, data)
    (N, _, n) = size(myAmica.models[1].source_signals)
    M = size(myAmica.models, 1)

    for h = 1:M
        mu = myAmica.models[h].location
        beta = myAmica.models[h].scale

        if myAmica.normalized_ica_weights[h] == 0
            continue
        end
        for i in 1:n
            tau = norm(myAmica.models[h].A[:, i])
            myAmica.models[h].A[:, i] = myAmica.models[h].A[:, i] / tau
            mu[:, i] = mu[:, i] * tau
            beta[:, i] = beta[:, i] / tau
        end

        if M > 1
            cnew = data * myAmica.ica_weights_per_sample[h, :] / (sum(myAmica.ica_weights_per_sample[h, :])) #todo: check why v not inverted
            for i in 1:n
                Wh = pinv(myAmica.models[h].A[:, :])
                mu[:, i] = mu[:, i] .- Wh[i, :]' * (cnew - myAmica.models[h].centers[:])
            end
            myAmica.models[h].centers = cnew
        end
        myAmica.models[h].location .= mu
        myAmica.models[h].scale .= beta
    end
end

# Sets the initial value for the shape parameter of the GeneralizedGaussians for each Model
function initialize_shape_parameter!(myAmica::SingleModelAmica, lrate::LearningRate)
    myAmica.shape .= lrate.shape0 .* myAmica.shape
end

function initialize_shape_parameter!(myAmica::MultiModelAmica, lrate::LearningRate)
    initialize_shape_parameter!.(myAmica.models, lrate)
end

function update_parameters!(myAmica::MultiModelAmica{T}, lrate::LearningRate, upd_shape::Bool) where {T<:Real}
    update_parameters!.(myAmica.models, lrate, upd_shape)
end

BLOCK_SIZE = 10_000

#Updates Gaussian mixture parameters. It also returns g, kappa and lamda which are needed to apply the newton method.
@views function update_parameters!(myAmica::SingleModelAmica{T}, data, lrate::LearningRate, upd_shape::Bool, newton_active::Bool) where {T<:Real}
    N, n, m = size(myAmica.y)
    num_blocks = cld(N, BLOCK_SIZE)

    # Initialize Lt with base values
    @timeit to "ldet" begin
        ldet = -logabsdet(myAmica.A |> Array)[1]
        myAmica.Lt .= ldet .+ myAmica.LLdetS
    end

    W = inv(myAmica.A)

    ArrayType = typeof(myAmica.shape)
    g_times_sources = zero(myAmica.dA)
    sum_z = zero(myAmica.shape)
    kp = zero(myAmica.shape)
    dmu_numer = zero(myAmica.shape)
    dmu_denom = zero(myAmica.shape)
    dbeta_denom = zero(myAmica.shape)
    dlambda_numer = zero(myAmica.shape)
    drho_numer = zero(myAmica.shape)

    for block in 1:num_blocks
        range = (1+((block-1)*BLOCK_SIZE)):(min(N, block * BLOCK_SIZE))

        # update sources
        myAmica.source_signals[range, :] .= data[range, :] * W'

        # calculate y
        for j in 1:m
            myAmica.y[range, :, j] .= myAmica.scale[:, j]' .* (myAmica.source_signals[range, :] .- myAmica.location[:, j]')
        end

        # calculate y_rho
        myAmica.y_rho[range, :, :] .= exp.((push_dimension(myAmica.shape .- T(1.0))) .* log.(abs.(notzero.(myAmica.y[range, :, :]))))


        @timeit to "qconst" QConst = .-log(T(2)) .- (loggamma.(T(1) .+ T(1) ./ myAmica.shape)) .+ log.(myAmica.proportions) .+ log.(myAmica.scale)
        # Q = qconst * abs(y)^rho 
        # Q = qconst * abs(y)^(rho - 1) * abs(y)
        @timeit to "Q" Q = push_dimension(QConst) .- (myAmica.y_rho[range, :, :] .* abs.(myAmica.y[range, :, :]))

        @timeit to "q_and_u" begin
            # Find max for numerical stability (over mixture components, dim 3)
            Qmax = maximum(Q, dims=3)

            # Compute logsumexp: Qmax + log(sum(exp(Q - Qmax)))
            logsumexp_Q = Qmax .+ log.(sum(exp.(Q .- Qmax), dims=3))

            # Compute z = exp(Q - logsumexp) + epsilon
            myAmica.z[range, :, :] .= exp.(Q .- logsumexp_Q) .+ T(1e-15)

            # Normalize z so that sum over mixtures = 1
            myAmica.z[range, :, :] ./= sum(myAmica.z[range, :, :], dims=3)

            # Accumulate Lt: sum logsumexp over channels (dim 2)
            myAmica.Lt[range] .+= dropdims(sum(logsumexp_Q, dims=(2, 3)), dims=(2, 3))
        end

        @timeit to "zfp/fp" begin
            fp = myAmica.y_rho[range, :, :] .* sign.(myAmica.y[range, :, :]) .* push_dimension(myAmica.shape)
            zfp = myAmica.z[range, :, :] .* fp
        end

        # g = sum(scale * z * fp, dims=3)
        # accumulate g' * source_signals
        @timeit to "dA" begin
            g_block = sum(push_dimension(myAmica.scale) .* zfp, dims=3)[:, :, 1]
            g_times_sources .+= g_block' * myAmica.source_signals[range, :]
        end

        # sum(z)
        @timeit to "sum_z" begin
            sum_z .+= sum(myAmica.z[range, :, :], dims=1)[1, :, :]
        end

        # sum(z * fp)
        @timeit to "dmu_numer" begin
            dmu_numer .+= sum(zfp, dims=1)[1, :, :]
        end

        # sum(z * fp * fp)
        @timeit to "kp" begin
            kp .+= sum(zfp .* fp, dims=1)[1, :, :]
        end

        # rho <= 2 ? sum(z * fp / y) : sum(z * fp * fp)
        @timeit to "dmu_denom" begin
            dmu_denom .+= sum(ifelse.(push_dimension(myAmica.shape) .<= T(2), zfp ./ notzero.(myAmica.y[range, :, :]), zfp .* fp), dims=1)[1, :, :] .* myAmica.scale
        end

        # rho <= 2.0 ? sum(z * fp * y) : 0
        @timeit to "dbeta_denom" begin
            dbeta_denom .+= sum(ifelse.(push_dimension(myAmica.shape) .<= T(2), zfp .* myAmica.y[range, :, :], T(0)), dims=1)[1, :, :]
        end

        # sum(z * (fp * y - 1)^2)
        @timeit to "dlambda_numer" begin
            dlambda_numer .+= sum(myAmica.z[range, :, :] .* (fp .* myAmica.y[range, :, :] .- T(1.0)) .^ 2, dims=1)[1, :, :]
        end

        # sum(z * log(y_rho * abs(y)) * y_rho * abs(y))
        @timeit to "drho_numer" begin
            y_rho_abs_y = myAmica.y_rho[range, :, :] .* abs.(myAmica.y[range, :, :])
            drho_numer .+= sum(ifelse.(y_rho_abs_y .>= T(1.0e-16), myAmica.z[range, :, :] .* log.(y_rho_abs_y) .* y_rho_abs_y, T(0.0)), dims=1)[1, :, :]
        end
    end

    # dA = I - g' * source_signals / N
    myAmica.dA .= ArrayType(I(n)) - g_times_sources / N
    @timeit to "myAmica.LL" push!(myAmica.LL, sum(myAmica.Lt) / (N * n))


    # alpha / proportions
    @timeit to "prop" if m > 1
        myAmica.proportions .= ifelse.(sum_z .>= T(0), sum_z ./ N, T(1) / N)
    end

    # newton parameters
    @timeit to "para" if newton_active
        dkap = @. (kp / (myAmica.proportions * N)) * myAmica.scale^2
        myAmica.newton_kappa .= sum(@. myAmica.proportions * dkap; dims=2)
        myAmica.newton_lambda .= sum(@. myAmica.proportions * (dlambda_numer / sum_z + dkap * myAmica.location^2); dims=2)
    end

    # mu / location
    @timeit to "loc" if m > 1
        myAmica.location .+= dmu_numer ./ dmu_denom
    end

    # sbeta / scale
    @timeit to "scale" begin
        # dbeta_numer / dbeta_denom
        myAmica.scale .*= sqrt.(sum_z ./ dbeta_denom)
    end

    # rho / shape
    @timeit to "shape" if upd_shape
        myAmica.shape .= clamp.(
            myAmica.shape .+ (lrate.shapelrate .* (1 .- (myAmica.shape ./ gpuDigamma.(1 .+ 1 ./ myAmica.shape)) .* drho_numer ./ sum_z)),
            lrate.minrho,
            lrate.maxrho
        )
    end

end

"updates the unmixed source_signals: myAmica.source_signals = myAmica.A ^ -1 * data"
function update_sources!(myAmica::SingleModelAmica{T}, data::AbstractMatrix{T}) where {T<:Real}
    W = inv(myAmica.A)
    myAmica.source_signals .= data * W'
end

function update_sources!(myAmica::MultiModelAmica, data)
    n = size(myAmica.models[1].A, 1)
    for h in 1:length(myAmica.models)
        for i in 1:n
            Wh = pinv(myAmica.models[h].A)
            myAmica.models[h].source_signals[i, :] = Wh[i, :]' * data .- Wh[i, :]' * myAmica.models[h].centers
        end
    end
end

#Adjusts learning rate depending on log-likelihood growth during past iterations. How many depends on iterwin. Uses LearningRate type from types.jl
function calculate_lrate!(
    myAmica::SingleModelAmica{T},
    iter::Int,
    newt_start_iter::Int,
    do_newton::Bool,
    lrate::LearningRate{T},
) where {T<:Real}
    # Check if likelihood is decreasing
    if myAmica.LL[iter] < myAmica.LL[iter-1]
        println("Likelihood decreasing!")

        # missing condition: .or. (ndtmpsum .le. min_nd)
        if lrate.lrate <= lrate.min
            println("minimum change threshold met, exiting loop ...")
            return true
        else
            # Decrease learning rates
            lrate.lrate *= lrate.lratefact
            lrate.shapelrate *= lrate.shapelratefact
            lrate.numdecs += 1

            if lrate.numdecs >= lrate.maxdecs
                lrate.lrate0 *= lrate.lratefact
                if iter > newt_start_iter
                    lrate.shapelrate0 *= lrate.shapelratefact
                end
                if do_newton && (iter > newt_start_iter)
                    println("Reducing maximum Newton lrate")
                    lrate.newtrate *= lrate.lratefact
                end
                lrate.numdecs = 0
            end
        end
    end

    false
end