"Process a range of blocks and accumulate results into acc for a specific thread"
@views function process_blocks!(
    myAmica::SingleModelAmica{T},
    data,
    W::AbstractMatrix{T},
    newton_active::Bool,
    tid::Int,
    num_blocks::Int;
    dump_dir::Union{Nothing,String}=nothing
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

        if !isnothing(dump_dir)
            write_binary(joinpath(dump_dir, "source_signals.bin"), source_signals)
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


        if !isnothing(dump_dir)
            write_binary(joinpath(dump_dir, "z.bin"), z)
            write_binary(joinpath(dump_dir, "y.bin"), y)
        end

        pool_release!("z", pool, z)
        pool_release!("y", pool, y)
    end


    merge!(to, lto, tree_point=["update_parameters"])
end
