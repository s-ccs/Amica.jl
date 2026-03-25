"""
    reparameterize!(myAmica::SingleModelAmica{T}) where T<:Real

Normalize mixing matrix columns and scale/location parameters accordingly.

Normalizes each column of the unmixing matrix A to unit norm and rescales the associated
scale and location parameters to maintain the same model representation. This reparameterization
ensures numerical stability and prevents the unmixing matrix from becoming ill-conditioned.

# Arguments
- `myAmica::SingleModelAmica{T}`: The AMICA model to reparameterize (modified in-place).

# Examples
```julia-repl
julia> reparameterize!(myAmica)
```

# See also
[`update_mixing!`](@ref)
"""
@views function reparameterize!(myAmica::SingleModelAmica{T}) where {T<:Real}
    # Calculate norm of column k: Anrmk = sqrt(sum(A(:,k)*A(:,k)))
    tau = sqrt.(sum(myAmica.A .^ 2, dims = 1))[1, :]

    mask = tau .> zero(T)
    myAmica.A .= ifelse.(push_dimension(mask), myAmica.A ./ tau', myAmica.A)
    myAmica.location .= ifelse.(mask, myAmica.location .* tau, myAmica.location)
    myAmica.scale .= ifelse.(mask, myAmica.scale ./ tau, myAmica.scale)
end

"""
    initialize_shape_parameter!(myAmica::SingleModelAmica, lrate::LearningRate)

Set initial shape parameters of generalized Gaussians for each component.

Initializes the shape parameter (rho) of the generalized Gaussian mixture components
by scaling the uniformly initialized shape values by the `shape0` factor from the learning rate.

# Arguments
- `myAmica::SingleModelAmica`: The AMICA model to initialize (modified in-place).
- `lrate::LearningRate`: Learning rate configuration containing the `shape0` scaling factor.

# Examples
```julia-repl
julia> initialize_shape_parameter!(myAmica, lrate)
```

# See also
[`amica!`](@ref), [`LearningRate`](@ref)
"""
# Sets the initial value for the shape parameter of the GeneralizedGaussians for each Model
@views function initialize_shape_parameter!(myAmica::SingleModelAmica, lrate::LearningRate)
    myAmica.shape .= lrate.shape0 .* myAmica.shape
end

"""
    update_parameters!(myAmica::SingleModelAmica{T}, data, lrate::LearningRate, upd_shape::Bool,
                       newton_active::Bool; dump_dir::Union{Nothing,String}=nothing) where T<:Real

Update all AMICA parameters from the data.

Processes the data in blocks to compute gradients and accumulators, then updates all model
parameters: mixture proportions (alpha), source location (mu), scale (beta), and optionally shape (rho).
Also updates Newton method parameters if Newton optimization is active.

# Arguments
- `myAmica::SingleModelAmica{T}`: The AMICA model to update (modified in-place).
- `data`: The input data matrix of shape (num_samples, num_features).
- `lrate::LearningRate`: Learning rate configuration.
- `upd_shape::Bool`: Whether to update shape parameters.
- `newton_active::Bool`: Whether Newton method parameters should be computed and updated.

# Keyword Arguments
- `dump_dir::Union{Nothing,String} = nothing`: Optional directory for debugging output.

# Examples
```julia-repl
julia> update_parameters!(myAmica, data, lrate, true, true)
```

# See also
[`process_blocks!`](@ref), [`update_mixing!`](@ref), [`reparameterize!`](@ref)
"""
#Updates Gaussian mixture parameters. It also returns g, kappa and lamda which are needed to apply the newton method.
@views function update_parameters!(
    myAmica::SingleModelAmica{T},
    data,
    lrate::LearningRate,
    upd_shape::Bool,
    newton_active::Bool;
    dump_dir::Union{Nothing,String} = nothing,
) where {T<:Real}
    N, n, m = myAmica.dims
    num_blocks = cld(N, myAmica.block_size)

    # Initialize Lt with base values
    @timeit_debug to "ldet" begin
        ldet = -logabsdet(myAmica.A |> Array)[1]
        myAmica.Lt .= ldet .+ myAmica.LLdetS
    end

    W = inv(myAmica.A)

    # Reset accumulator for this iteration
    reset!(myAmica.acc)

    @timeit_debug to "process_blocks" if myAmica.num_threads == 1
        # Single-threaded path
        process_blocks!(myAmica, data, W, newton_active, 1, num_blocks; dump_dir)
    else
        # Multi-threaded path: divide blocks among threads
        Threads.@threads for tid = 1:min(myAmica.num_threads, num_blocks)
            process_blocks!(myAmica, data, W, newton_active, tid, num_blocks; dump_dir)
        end
    end

    # Extract final accumulated values (from first thread slot after reduction)
    @timeit_debug to "accumulate" begin
        g_times_sources = sum(myAmica.acc.g_times_sources, dims = 3)[:, :, 1]
        sum_z = sum(myAmica.acc.sum_z, dims = 3)[:, :, 1]
        kp = sum(myAmica.acc.kp, dims = 3)[:, :, 1]
        dmu_numer = sum(myAmica.acc.dmu_numer, dims = 3)[:, :, 1]
        dmu_denom = sum(myAmica.acc.dmu_denom, dims = 3)[:, :, 1]
        dbeta_denom = sum(myAmica.acc.dbeta_denom, dims = 3)[:, :, 1]
        dlambda_numer = sum(myAmica.acc.dlambda_numer, dims = 3)[:, :, 1]
        drho_numer = sum(myAmica.acc.drho_numer, dims = 3)[:, :, 1]
        myAmica.newton_sigma2 .= sum(myAmica.acc.newton_sigma2, dims = 2)[:, 1]
        myAmica.Lt .+= sum(myAmica.acc.Lt_accum, dims = 2)[:, 1]
    end

    # dA = I - g' * source_signals / N
    @timeit_debug to "myAmica.dA" myAmica.dA .=
        (I(n) |> typeof(myAmica.A)) - g_times_sources / N
    @timeit_debug to "myAmica.LL" push!(myAmica.LL, sum(myAmica.Lt) / (N * n))


    # alpha / proportions
    @timeit_debug to "prop" if m > 1
        myAmica.proportions .= ifelse.(sum_z .>= T(0), sum_z ./ N, T(1) / N)
    end

    # newton parameters
    @timeit_debug to "para" if newton_active
        dkap = @. (kp / (myAmica.proportions * N)) * myAmica.scale^2
        myAmica.newton_kappa .= sum(@. myAmica.proportions * dkap; dims = 2)
        myAmica.newton_lambda .= sum(
            @. myAmica.proportions * (dlambda_numer / sum_z + dkap * myAmica.location^2);
            dims = 2,
        )
    end

    # mu / location
    @timeit_debug to "loc" if m > 1
        myAmica.location .+= dmu_numer ./ dmu_denom
    end

    # sbeta / scale
    @timeit_debug to "scale" begin
        # dbeta_numer / dbeta_denom
        myAmica.scale .*= sqrt.(sum_z ./ dbeta_denom)
    end

    # rho / shape
    @timeit_debug to "shape" if upd_shape
        myAmica.shape .=
            clamp.(
                myAmica.shape .+ (
                    lrate.shapelrate .* (
                        1 .-
                        (myAmica.shape ./ gpuDigamma.(1 .+ 1 ./ myAmica.shape)) .*
                        drho_numer ./ sum_z
                    )
                ),
                lrate.minrho,
                lrate.maxrho,
            )
    end

end
