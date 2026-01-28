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
    N, n, m = myAmica.dims
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
    myAmica.newton_sigma2 .= zero(T)

    for block in 1:num_blocks
        lower = (1 + ((block - 1) * BLOCK_SIZE))
        upper = (min(N, block * BLOCK_SIZE))
        full_range = lower:upper
        n_samples = upper - lower + 1
        r = 1:(n_samples)

        source_signals = pool_acquire!(myAmica.pool, (n_samples, n))
        # update sources
        source_signals .= data[full_range, :] * W'

        y = pool_acquire!(myAmica.pool, (n_samples, n, m))

        # calculate y 
        for j in 1:m
            y[:, :, j] .= myAmica.scale[:, j]' .* (source_signals .- myAmica.location[:, j]')
        end

        y_rho = pool_acquire!(myAmica.pool, (n_samples, n, m))

        # calculate y_rho
        y_rho .= exp.((push_dimension(myAmica.shape .- T(1.0))) .* log.(abs.(notzero.(y))))

        @timeit to "q_and_u" begin
            # Q = qconst * abs(y)^rho 
            # Q = qconst * abs(y)^(rho - 1) * abs(y)
            @timeit to "qconst" QConst = .-log(T(2)) .- (loggamma.(T(1) .+ T(1) ./ myAmica.shape)) .+ log.(myAmica.proportions) .+ log.(myAmica.scale)

            Q = pool_acquire!(myAmica.pool, (n_samples, n, m))

            @timeit to "Q" Q .= push_dimension(QConst) .- (y_rho .* abs.(y))

            @timeit to "logexp" begin
                # Find max for numerical stability (over mixture components, dim 3)
                maximum!(myAmica.scratch2[r, :, 1:1], Q)
                # Compute logsumexp: Qmax + log(sum(exp(Q - Qmax)))
                myAmica.scratch2[r, :, 1] .= myAmica.scratch2[r, :, 1:1] .+ log.(sum(exp.(Q .- myAmica.scratch2[r, :, 1:1]), dims=3))
            end

            # Compute z = exp(Q - logsumexp) + epsilon
            z = pool_acquire!(myAmica.pool, (n_samples, n, m))

            @timeit to "z" z .= exp.(Q .- myAmica.scratch2[r, :, 1]) .+ T(1e-15)
            pool_release!(myAmica.pool, Q)

            # Accumulate Lt: sum logsumexp over channels (dim 2)
            @timeit to "Lt" begin
                sum!(myAmica.scratch1[r, 1:1, 1], myAmica.scratch2[r, :, 1])
                myAmica.Lt[full_range] .+= myAmica.scratch1[r, 1, 1]
            end

            # scratch1 = sum(z)
            # Normalize z so that sum over mixtures = 1
            @timeit to "z_norm" begin

                sum!(myAmica.scratch1[r, :, 1], z)
                z ./= myAmica.scratch1[r, :, 1]
            end
        end

        @timeit to "fp" begin
            fp = pool_acquire!(myAmica.pool, (n_samples, n, m))
            fp .= y_rho .* sign.(y) .* push_dimension(myAmica.shape)
        end
        pool_release!(myAmica.pool, y_rho)

        # g = sum(scale * z * fp, dims=3)
        # accumulate g' * source_signals
        @timeit to "dA" begin
            g_block = sum(push_dimension(myAmica.scale) .* z .* fp, dims=3)[:, :, 1]
            g_times_sources .+= g_block' * source_signals
        end

        pool_release!(myAmica.pool, source_signals)

        # sum(z)
        @timeit to "sum_z" begin
            sum_z .+= sum(z, dims=1)[1, :, :]
        end

        # sum(z * fp)
        @timeit to "dmu_numer" begin
            dmu_numer .+= sum(fp .* z, dims=1)[1, :, :]
        end

        # sum(z * fp * fp)
        @timeit to "kp" begin
            kp .+= sum(fp .* z .* fp, dims=1)[1, :, :]
        end

        # rho <= 2 ? sum(z * fp / y) : sum(z * fp * fp)
        @timeit to "dmu_denom" begin
            myAmica.scratch2[r, :, :] .= ifelse.(
                push_dimension(myAmica.shape) .<= T(2),
                z .* fp ./ notzero.(y),
                z .* fp .* fp)

            sum!(myAmica.scratch2[1:1, :, :], myAmica.scratch2[r, :, :])
            dmu_denom .+= myAmica.scratch2[1, :, :] .* myAmica.scale
        end

        # rho <= 2.0 ? sum(z * fp * y) : 0
        @timeit to "dbeta_denom" begin
            myAmica.scratch2[r, :, :] .= ifelse.(push_dimension(myAmica.shape) .<= T(2), fp .* z .* y, T(0))
            sum!(myAmica.scratch2[1:1, :, :], myAmica.scratch2[r, :, :])
            dbeta_denom .+= myAmica.scratch2[1, :, :]
        end

        # sum(z * (fp * y - 1)^2)
        @timeit to "dlambda_numer" begin
            myAmica.scratch2[r, :, :] .= z .* (fp .* y .- T(1.0)) .^ 2
            pool_release!(myAmica.pool, fp)

            sum!(myAmica.scratch2[1:1, :, :], myAmica.scratch2[r, :, :])
            dlambda_numer .+= myAmica.scratch2[1, :, :]
        end

        # sum(z * log(y_rho * abs(y)) * y_rho * abs(y))
        @timeit to "drho_numer" begin
            myAmica.scratch2[r, :, :] .= y_rho .* abs.(y)
            myAmica.scratch2[r, :, :] .= ifelse.(
                myAmica.scratch2[r, :, :] .>= T(1.0e-16),
                z .* log.(myAmica.scratch2[r, :, :]) .* myAmica.scratch2[r, :, :],
                T(0.0)
            )
            sum!(myAmica.scratch2[1:1, :, :], myAmica.scratch2[r, :, :])
            drho_numer .+= myAmica.scratch2[1, :, :]
        end

        pool_release!(myAmica.pool, z)
        pool_release!(myAmica.pool, y)

        @timeit to "newton_sigma2" if newton_active
            myAmica.newton_sigma2 .+= sum(source_signals .^ 2, dims=1)[1, :] ./ N
        end
    end

    # dA = I - g' * source_signals / N
    @timeit to "myAmica.dA" myAmica.dA .= ArrayType(I(n)) - g_times_sources / N
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