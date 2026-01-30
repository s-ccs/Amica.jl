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


#Updates Gaussian mixture parameters. It also returns g, kappa and lamda which are needed to apply the newton method.
@views function update_parameters!(myAmica::SingleModelAmica{T}, data, lrate::LearningRate, upd_shape::Bool, newton_active::Bool) where {T<:Real}
    N, n, m = myAmica.dims
    num_blocks = cld(N, myAmica.block_size)

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
        lower = (1 + ((block - 1) * myAmica.block_size))
        upper = (min(N, block * myAmica.block_size))
        full_range = lower:upper
        n_samples = upper - lower + 1


        source_signals = pool_acquire!("source_signals", myAmica.pool, (n_samples, n))
        # update sources
        @timeit to "source_signals" mul!(source_signals, data[full_range, :], W')


        @timeit to "y" begin
            y = pool_acquire!("y", myAmica.pool, (n_samples, n, m))

            # calculate y 
            for j in 1:m
                y[:, :, j] .= myAmica.scale[:, j]' .* (source_signals .- myAmica.location[:, j]')
            end
        end


        @timeit to "y_rho" begin
            y_rho = pool_acquire!("y_rho", myAmica.pool, (n_samples, n, m))

            # calculate y_rho
            y_rho .= exp.((push_dimension(myAmica.shape .- T(1.0))) .* log.(abs.(notzero.(y))))
        end

        @timeit to "q_and_u" begin
            # Q = qconst * abs(y)^rho 
            # Q = qconst * abs(y)^(rho - 1) * abs(y)
            @timeit to "qconst" QConst = .-log(T(2)) .- (loggamma.(T(1) .+ T(1) ./ myAmica.shape)) .+ log.(myAmica.proportions) .+ log.(myAmica.scale)

            Q = pool_acquire!("Q", myAmica.pool, (n_samples, n, m))

            @timeit to "Q" Q .= push_dimension(QConst) .- (y_rho .* abs.(y))

            @timeit to "logexp" begin
                # Find max for numerical stability (over mixture components, dim 3)
                Qmax = pool_acquire!("Qmax", myAmica.pool, (n_samples, n, 1))
                @timeit to "Qmax" maximum!(Qmax, Q)
                # Compute logsumexp: Qmax + log(sum(exp(Q - Qmax)))
                expQ = pool_acquire!("expQ", myAmica.pool, (n_samples, n, m))

                @timeit to "expQ" expQ .= exp.(Q .- Qmax)

                logexp = pool_acquire!("logexp", myAmica.pool, (n_samples, n, 1))

                @timeit to "sum" sum!(logexp, expQ)
                pool_release!("expQ", myAmica.pool, expQ)

                @timeit to "calc" begin
                    logexp .= log.(logexp) .+ Qmax
                end
                pool_release!("Qmax", myAmica.pool, Qmax)
            end


            # Compute z = exp(Q - logsumexp) + epsilon
            z = pool_acquire!("z", myAmica.pool, (n_samples, n, m))

            @timeit to "z" z .= exp.(Q .- logexp) .+ T(1e-15)
            pool_release!("Q", myAmica.pool, Q)

            # Accumulate Lt: sum logsumexp over channels (dim 2)
            @timeit to "Lt" begin
                sum_logexp = pool_acquire!("sum_logexp", myAmica.pool, (n_samples, 1, 1))

                sum!(sum_logexp, logexp)
                pool_release!("logexp", myAmica.pool, logexp)

                myAmica.Lt[full_range] .+= sum_logexp
                pool_release!("sum_logexp", myAmica.pool, sum_logexp)
            end

            # Normalize z so that sum over mixtures = 1
            @timeit to "z_norm" begin
                zsum = pool_acquire!("zsum", myAmica.pool, (n_samples, n, 1))
                sum!(zsum, z)
                z ./= zsum
                pool_release!("zsum", myAmica.pool, zsum)
            end
        end

        @timeit to "fp" begin
            fp = pool_acquire!("fp", myAmica.pool, (n_samples, n, m))
            fp .= y_rho .* sign.(y) .* push_dimension(myAmica.shape)
        end

        # sum(z * log(y_rho * abs(y)) * y_rho * abs(y))
        @timeit to "drho_numer" begin
            scratch = pool_acquire!("scratch", myAmica.pool, (n_samples, n, m))
            scratch .= y_rho .* abs.(y)
            scratch .= ifelse.(
                scratch .>= T(1.0e-16),
                z .* log.(scratch) .* scratch,
                T(0.0)
            )
            drho_numer .+= sum(scratch, dims=1)[1, :, :]
            pool_release!("scratch", myAmica.pool, scratch)
        end
        pool_release!("y_rho", myAmica.pool, y_rho)

        # g = sum(scale * z * fp, dims=3)
        # accumulate g' * source_signals
        @timeit to "dA" begin
            g_block = pool_acquire!("g_block", myAmica.pool, (n_samples, n, m))
            g_block_sum = pool_acquire!("g_block_sum", myAmica.pool, (n_samples, n, 1))

            g_block .= push_dimension(myAmica.scale) .* z .* fp
            sum!(g_block_sum, g_block)
            g_times_sources .+= g_block_sum[:, :, 1]' * source_signals
            pool_release!("g_block", myAmica.pool, g_block)
            pool_release!("g_block_sum", myAmica.pool, g_block_sum)
        end

        @timeit to "newton_sigma2" if newton_active
            scratch = pool_acquire!("scratch", myAmica.pool, (n_samples, n))
            scratch .= source_signals .^ 2
            myAmica.newton_sigma2 .+= sum(scratch, dims=1)[1, :] ./ N
            pool_release!("scratch", myAmica.pool, scratch)
        end

        pool_release!("source_signals", myAmica.pool, source_signals)

        # sum(z)
        @timeit to "sum_z" begin
            sum_z .+= sum(z, dims=1)[1, :, :]
        end

        # sum(z * fp)
        @timeit to "dmu_numer" begin
            scratch = pool_acquire!("scratch", myAmica.pool, (n_samples, n, m))
            scratch .= fp .* z
            dmu_numer .+= sum(scratch, dims=1)[1, :, :]
            pool_release!("scratch", myAmica.pool, scratch)
        end

        # sum(z * fp * fp)
        @timeit to "kp" begin
            scratch = pool_acquire!("scratch", myAmica.pool, (n_samples, n, m))
            scratch .= fp .* z .* fp
            kp .+= sum(scratch, dims=1)[1, :, :]

            pool_release!("scratch", myAmica.pool, scratch)

        end

        # rho <= 2 ? sum(z * fp / y) : sum(z * fp * fp)
        @timeit to "dmu_denom" begin
            scratch = pool_acquire!("scratch", myAmica.pool, (n_samples, n, m))
            scratch .= ifelse.(
                push_dimension(myAmica.shape) .<= T(2),
                z .* fp ./ notzero.(y),
                z .* fp .* fp)


            dmu_denom .+= sum(scratch, dims=1)[1, :, :] .* myAmica.scale
            pool_release!("scratch", myAmica.pool, scratch)
        end

        # rho <= 2.0 ? sum(z * fp * y) : 0
        @timeit to "dbeta_denom" begin
            scratch = pool_acquire!("scratch", myAmica.pool, (n_samples, n, m))
            scratch .= ifelse.(push_dimension(myAmica.shape) .<= T(2), fp .* z .* y, T(0))
            dbeta_denom .+= sum(scratch, dims=1)[1, :, :]
            pool_release!("scratch", myAmica.pool, scratch)
        end

        # sum(z * (fp * y - 1)^2)
        @timeit to "dlambda_numer" begin
            scratch = pool_acquire!("scratch", myAmica.pool, (n_samples, n, m))
            scratch .= z .* (fp .* y .- T(1.0)) .^ 2
            pool_release!("fp", myAmica.pool, fp)

            dlambda_numer .+= sum(scratch, dims=1)[1, :, :]
            pool_release!("scratch", myAmica.pool, scratch)
        end

        pool_release!("z", myAmica.pool, z)
        pool_release!("y", myAmica.pool, y)
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