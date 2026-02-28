"Normalizes source density location parameter (mu), scale parameter (beta) and model centers"
function reparameterize!(myAmica::SingleModelAmica{T}) where T<:Real
    # Calculate norm of column k: Anrmk = sqrt(sum(A(:,k)*A(:,k)))
    tau = sqrt.(sum(myAmica.A .^ 2, dims=1))[1, :]

    mask = tau .> zero(T)
    myAmica.A .= ifelse.(push_dimension(mask), myAmica.A ./ tau', myAmica.A)
    myAmica.location .= ifelse.(mask, myAmica.location .* tau, myAmica.location)
    myAmica.scale .= ifelse.(mask, myAmica.scale ./ tau, myAmica.scale)
end

# Sets the initial value for the shape parameter of the GeneralizedGaussians for each Model
function initialize_shape_parameter!(myAmica::SingleModelAmica, lrate::LearningRate)
    myAmica.shape .= lrate.shape0 .* myAmica.shape
end

"Process a range of blocks and accumulate results into acc for a specific thread"
@views function process_blocks!(
    myAmica::SingleModelAmica{T},
    data,
    W::AbstractMatrix{T},
    newton_active::Bool,
    tid::Int,
    num_blocks::Int
) where {T<:Real}
    N, n, m = myAmica.dims

    lto = TimerOutput()

    blocks_per_thread = cld(num_blocks, myAmica.num_threads)
    start_block = (tid - 1) * blocks_per_thread + 1
    end_block = min(tid * blocks_per_thread, num_blocks)

    pool = myAmica.pools[tid]

    # Thread-local views into the accumulator arrays
    g_times_sources_t = myAmica.acc.g_times_sources[:, :, tid]
    sum_z_t = myAmica.acc.sum_z[:, :, tid]
    kp_t = myAmica.acc.kp[:, :, tid]
    dmu_numer_t = myAmica.acc.dmu_numer[:, :, tid]
    dmu_denom_t = myAmica.acc.dmu_denom[:, :, tid]
    dbeta_denom_t = myAmica.acc.dbeta_denom[:, :, tid]
    dlambda_numer_t = myAmica.acc.dlambda_numer[:, :, tid]
    drho_numer_t = myAmica.acc.drho_numer[:, :, tid]
    newton_sigma2_t = myAmica.acc.newton_sigma2[:, tid]
    Lt_accum_t = myAmica.acc.Lt_accum[:, tid]

    for block in start_block:end_block
        lower = (1 + ((block - 1) * myAmica.block_size))
        upper = (min(N, block * myAmica.block_size))
        full_range = lower:upper
        n_samples = upper - lower + 1

        source_signals = pool_acquire!("source_signals", pool, (n_samples, n))
        # update sources
        @timeit lto "source_signals" mul!(source_signals, data[full_range, :], W')

        @timeit lto "y" begin
            y = pool_acquire!("y", pool, (n_samples, n, m))

            # calculate y 
            for j in 1:m
                y[:, :, j] .= myAmica.scale[:, j]' .* (source_signals .- myAmica.location[:, j]')
            end
        end

        @timeit lto "y_rho" begin
            y_rho = pool_acquire!("y_rho", pool, (n_samples, n, m))

            # calculate y_rho
            y_rho .= exp.((push_dimension(myAmica.shape .- T(1.0))) .* log.(abs.(notzero.(y))))
        end

        # q and u
        begin
            # Q = qconst * abs(y)^rho 
            # Q = qconst * abs(y)^(rho - 1) * abs(y)
            @timeit lto "qconst" QConst = .-log(T(2)) .- (loggamma.(T(1) .+ T(1) ./ myAmica.shape)) .+ log.(myAmica.proportions) .+ log.(myAmica.scale)

            Q = pool_acquire!("Q", pool, (n_samples, n, m))

            @timeit lto "Q" Q .= push_dimension(QConst) .- (y_rho .* abs.(y))

            # compute logexp(Q)
            begin
                # Find max for numerical stability (over mixture components, dim 3)
                Qmax = pool_acquire!("Qmax", pool, (n_samples, n, 1))
                @timeit lto "Qmax" maximum!(Qmax, Q)
                # Compute logsumexp: Qmax + log(sum(exp(Q - Qmax)))
                expQ = pool_acquire!("expQ", pool, (n_samples, n, m))

                @timeit lto "expQ" expQ .= exp.(Q .- Qmax)

                logexp = pool_acquire!("logexp", pool, (n_samples, n, 1))

                @timeit lto "sum" sum!(logexp, expQ)
                pool_release!("expQ", pool, expQ)

                @timeit lto "calc" begin
                    logexp .= log.(logexp) .+ Qmax
                end
                pool_release!("Qmax", pool, Qmax)
            end

            # Compute z = exp(Q - logsumexp) + epsilon
            z = pool_acquire!("z", pool, (n_samples, n, m))

            @timeit lto "z" z .= exp.(Q .- logexp) .+ T(1e-15)
            pool_release!("Q", pool, Q)

            # Accumulate Lt: sum logsumexp over channels (dim 2)
            @timeit lto "Lt" begin
                sum_logexp = pool_acquire!("sum_logexp", pool, (n_samples, 1, 1))

                sum!(sum_logexp, logexp)
                pool_release!("logexp", pool, logexp)

                Lt_accum_t[full_range] .+= sum_logexp
                pool_release!("sum_logexp", pool, sum_logexp)
            end

            # Normalize z so that sum over mixtures = 1
            @timeit lto "z_norm" begin
                zsum = pool_acquire!("zsum", pool, (n_samples, n, 1))
                sum!(zsum, z)
                z ./= zsum
                pool_release!("zsum", pool, zsum)
            end
        end

        @timeit lto "fp" begin
            fp = pool_acquire!("fp", pool, (n_samples, n, m))
            fp .= y_rho .* sign.(y) .* push_dimension(myAmica.shape)
        end

        # sum(z * log(y_rho * abs(y)) * y_rho * abs(y))
        @timeit lto "drho_numer" begin
            scratch = pool_acquire!("scratch", pool, (n_samples, n, m))
            scratch .= y_rho .* abs.(y)
            scratch .= ifelse.(
                scratch .>= T(1.0e-16),
                z .* log.(scratch) .* scratch,
                T(0.0)
            )
            drho_numer_t .+= sum(scratch, dims=1)[1, :, :]
            pool_release!("scratch", pool, scratch)
        end
        pool_release!("y_rho", pool, y_rho)

        # g = sum(scale * z * fp, dims=3)
        # accumulate g' * source_signals
        @timeit lto "dA" begin
            g_block = pool_acquire!("g_block", pool, (n_samples, n, m))
            g_block_sum = pool_acquire!("g_block_sum", pool, (n_samples, n, 1))

            g_block .= push_dimension(myAmica.scale) .* z .* fp
            sum!(g_block_sum, g_block)
            g_times_sources_t .+= g_block_sum[:, :, 1]' * source_signals
            pool_release!("g_block", pool, g_block)
            pool_release!("g_block_sum", pool, g_block_sum)
        end

        @timeit lto "newton_sigma2" if newton_active
            scratch = pool_acquire!("scratch2", pool, (n_samples, n))
            scratch .= source_signals .^ 2
            newton_sigma2_t .+= sum(scratch, dims=1)[1, :] ./ N
            pool_release!("scratch2", pool, scratch)
        end

        pool_release!("source_signals", pool, source_signals)

        # sum(z)
        @timeit lto "sum_z" begin
            sum_z_t .+= sum(z, dims=1)[1, :, :]
        end

        # sum(z * fp)
        @timeit lto "dmu_numer" begin
            scratch = pool_acquire!("scratch", pool, (n_samples, n, m))
            scratch .= fp .* z
            dmu_numer_t .+= sum(scratch, dims=1)[1, :, :]
            pool_release!("scratch", pool, scratch)
        end

        # sum(z * fp * fp)
        @timeit lto "kp" begin
            scratch = pool_acquire!("scratch", pool, (n_samples, n, m))
            scratch .= fp .* z .* fp
            kp_t .+= sum(scratch, dims=1)[1, :, :]

            pool_release!("scratch", pool, scratch)
        end

        # rho <= 2 ? sum(z * fp / y) : sum(z * fp * fp)
        @timeit lto "dmu_denom" begin
            scratch = pool_acquire!("scratch", pool, (n_samples, n, m))
            scratch .= ifelse.(
                push_dimension(myAmica.shape) .<= T(2),
                z .* fp ./ notzero.(y),
                z .* fp .* fp)


            dmu_denom_t .+= sum(scratch, dims=1)[1, :, :] .* myAmica.scale
            pool_release!("scratch", pool, scratch)
        end

        # rho <= 2.0 ? sum(z * fp * y) : 0
        @timeit lto "dbeta_denom" begin
            scratch = pool_acquire!("scratch", pool, (n_samples, n, m))
            scratch .= ifelse.(push_dimension(myAmica.shape) .<= T(2), fp .* z .* y, T(0))
            dbeta_denom_t .+= sum(scratch, dims=1)[1, :, :]
            pool_release!("scratch", pool, scratch)
        end

        # sum(z * (fp * y - 1)^2)
        @timeit lto "dlambda_numer" begin
            scratch = pool_acquire!("scratch", pool, (n_samples, n, m))
            scratch .= z .* (fp .* y .- T(1.0)) .^ 2
            pool_release!("fp", pool, fp)

            dlambda_numer_t .+= sum(scratch, dims=1)[1, :, :]
            pool_release!("scratch", pool, scratch)
        end

        pool_release!("z", pool, z)
        pool_release!("y", pool, y)
    end


    merge!(to, lto, tree_point=["update_parameters"])
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

    # Reset accumulator for this iteration
    reset!(myAmica.acc)

    if myAmica.num_threads == 1
        # Single-threaded path
        process_blocks!(myAmica, data, W, newton_active, 1, num_blocks)
    else
        # Multi-threaded path: divide blocks among threads
        Threads.@threads for tid in 1:min(myAmica.num_threads, num_blocks)
            process_blocks!(myAmica, data, W, newton_active, tid, num_blocks)
        end
    end

    # Extract final accumulated values (from first thread slot after reduction)
    @timeit to "accumulate" begin
        g_times_sources = sum(myAmica.acc.g_times_sources, dims=3)[:, :, 1]
        sum_z = sum(myAmica.acc.sum_z, dims=3)[:, :, 1]
        kp = sum(myAmica.acc.kp, dims=3)[:, :, 1]
        dmu_numer = sum(myAmica.acc.dmu_numer, dims=3)[:, :, 1]
        dmu_denom = sum(myAmica.acc.dmu_denom, dims=3)[:, :, 1]
        dbeta_denom = sum(myAmica.acc.dbeta_denom, dims=3)[:, :, 1]
        dlambda_numer = sum(myAmica.acc.dlambda_numer, dims=3)[:, :, 1]
        drho_numer = sum(myAmica.acc.drho_numer, dims=3)[:, :, 1]
        myAmica.newton_sigma2 .= sum(myAmica.acc.newton_sigma2, dims=2)[:, 1]
        myAmica.Lt .+= sum(myAmica.acc.Lt_accum, dims=2)[:, 1]
    end

    # dA = I - g' * source_signals / N
    ArrayType = typeof(myAmica.shape)
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