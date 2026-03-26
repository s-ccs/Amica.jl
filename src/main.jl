"""
    fit(amicaType::Type{AmicaKind}, data::Array{T,2}; m=3, maxiter=500, location=nothing,
        scale=nothing, A=nothing, ArrayType=Array, block_size=10_000, num_threads=1, kwargs...)

Fit AMICA model to data and return the fitted model.

Performs Independent Component Analysis using the Adaptive Mixture ICA (AMICA) algorithm.
Creates an AMICA model of the specified type, initializes it with the provided or default parameters,
and fits it to the input data using the `amica!()` function.

# Arguments
- `amicaType::Type{AmicaKind}`: The type of AMICA model to create (e.g., `SingleModelAmica`).
- `data::Array{T,2}`: Input data matrix of shape (num_samples, num_channels).

# Keyword Arguments
- `m::Int = 3`: Number of mixture components in the generalized Gaussian model.
- `maxiter::Int = 500`: Maximum number of iterations.
- `location::Union{Nothing, Array} = nothing`: Initial location parameters. If `nothing`, initialized automatically.
- `scale::Union{Nothing, Array} = nothing`: Initial scale parameters. If `nothing`, initialized automatically.
- `A::Union{Nothing, Array} = nothing`: Initial unmixing matrix. If `nothing`, initialized as random near-identity.
- `ArrayType::Type{<:DenseArray} = Array`: Array type to use (e.g., `Array` for CPU, `CuArray` for GPU).
- `block_size::Int = 10_000`: Number of samples to process in each block for memory efficiency.
- `num_threads::Int = 1`: Number of threads for parallel block processing.
- `do_fit::Bool = true`: Whether to perform fitting immediately. If `false`, returns uninitialized model. Use amica!(myAmica,data) to fit
- `sort_by_variance::Bool = false`: Whether to sort the recovered sources by variance after fitting.
- `kwargs...`: Additional keyword arguments passed to `amica!()` (e.g., `lrate`, `do_sphering`, `remove_mean`,`show_progress`,`show_timing``).

# Returns
- `model::AmicaKind`: Fitted AMICA model object with learned parameters.

# Examples
```julia-repl
julia> data = randn(10000, 32)  # 10000 samples, 32 components
julia> model = fit(SingleModelAmica, data; m=3, maxiter=100)
julia> typeof(model)
SingleModelAmica{Float64, Vector{Float64}, Matrix{Float64}, Array{Float64, 3}}

julia> model = fit(SingleModelAmica, data; m=4, maxiter=200, num_threads=4)
```

# See also
[`amica!`](@ref), [`SingleModelAmica`](@ref), [`recover_sources`](@ref)
"""
function StatsAPI.fit(
    amicaType::Type{AmicaKind},
    data::ArrayType;
    m = 3,
    maxiter = 500,
    location = nothing,
    scale = nothing,
    A = nothing,
    #ArrayType::Type{<:DenseArray} = Array,
    block_size = 10_000,
    num_threads = 1,
    do_fit = true,
    kwargs...,
) where {AmicaKind<:AbstractAmica,ArrayType<:AbstractArray}
    (N, n) = size(data)
    amica = AmicaKind(
        data,
        m = m,
        ncomps = n,
        nsamples = N,
        location = location,
        scale = scale,
        A = A,
        block_size = block_size,
        num_threads = num_threads,
    )
    if do_fit
        amica!(amica, data; maxiter, kwargs...)
    end
    return amica
end

"""
    materialize_data_like(target::AbstractMatrix{T}, data::AbstractMatrix{T})::AbstractMatrix{T}
    where T<:Real

Convert data format to match target array type (e.g., CPU to GPU array).

This function replicates the structure and type of the `target` array for the `data` array,
making it useful for converting between different array backends (e.g., from regular CPU arrays
to GPU arrays or vice versa) while preserving data type and dimensions.

# Arguments
- `target::AbstractMatrix{T}`: A matrix of the target type to match (determines array backend).
- `data::AbstractMatrix{T}`: A matrix containing the data to convert, with same element type as target.

# Returns
- `result::AbstractMatrix{T}`: A new matrix with data copied to the target array type/backend.

# Examples
```julia-repl
julia> cpu_data = [1.0 2.0; 3.0 4.0]
julia> gpu_target = CUDA.CuArray([0.0 0.0; 0.0 0.0])
julia> gpu_data = materialize_data_like(gpu_target, cpu_data)
julia> typeof(gpu_data)
CUDA.CuArray{Float64, 2}
```
"""
function materialize_data_like(
    target::AbstractMatrix{T},
    data::AbstractMatrix{T},
) where {T<:Real}
    if data isa typeof(target)
        return copy(data)
    end

    host_data = Matrix{T}(undef, size(data)...)
    copyto!(host_data, data)

    out = similar(target, size(data))
    copyto!(out, host_data)
    return out
end

"""
    amica!(myAmica, data; lrate=LearningRate(), remove_mean=true, do_sphering=true,
           show_progress=true, maxiter=50, do_newton=true, newt_start_iter=50,
           iterwin=10, update_shape=true, data_inplace=false, mindll=1e-8,
           dump_dir=nothing, show_timing=false)

Fit `myAmica` on `data` and return the model.
"""
@views function amica!(
    myAmica::AbstractAmica,
    data::AbstractMatrix{T};
    lrate::LearningRate{T} = LearningRate{T}(),
    remove_mean::Bool = true,
    do_sphering::Bool = true,
    show_progress::Bool = true,
    maxiter::Int = 50,
    do_newton::Bool = true,
    newt_start_iter::Int = 50,
    iterwin::Int = 10,
    update_shape::Bool = true,
    data_inplace::Bool = false,
    mindll::T = T(1e-8),
    dump_dir::Union{Nothing,String} = nothing,
    sort_by_variance = false,
    show_timing = false,
) where {T<:Real}

    working_data = data_inplace ? data : copy(data)

    @timeit_debug to "initialize_shape_parameter!" initialize_shape_parameter!(
        myAmica,
        lrate,
    )
    myAmica.no_newton = false

    #Prepares data by removing means and/or sphering
    if remove_mean
        @timeit_debug to "removeMean" removed_mean = remove_mean!(working_data)
    end

    @timeit_debug to "sphering" if do_sphering
        S = sphering!(working_data)
        myAmica.S = S
        myAmica.LLdetS = logabsdet(S |> Array)[1]
    else
        myAmica.S = I
        myAmica.LLdetS = 0
    end

    dLL = zeros(1, maxiter)

    @timeit_debug to "materialize_data" data =
        materialize_data_like(myAmica.A, working_data)


    niter = 0


    p = ProgressUnknown(; enabled = show_progress, showspeed = true, spinner = true)

    for iter = 1:maxiter
        niter += 1

        @timeit_debug to "update_parameters" begin
            update_parameters!(
                myAmica,
                data,
                lrate,
                update_shape,
                do_newton && iter >= newt_start_iter;
                dump_dir,
            )
        end

        @timeit_debug to "calculate_DLL" begin
            calculate_DLL!(dLL, myAmica, iter)
        end

        @timeit_debug to "update_mixing" begin
            update_mixing!(myAmica, iter, do_newton, newt_start_iter, lrate)
        end

        @timeit_debug to "reparameterize" begin
            reparameterize!(myAmica)
        end

        if iter > 1
            # Check for NaN
            if isnan(myAmica.LL[iter])
                @warn("Got NaN! Exiting ...")
                break
            end

            if calculate_lrate!(myAmica, iter, newt_start_iter, do_newton, lrate)
                break
            end

            # Checks termination criterion
            if iter > iterwin
                sdll = sum(dLL[iter-iterwin+1:iter]) / iterwin
                if (sdll > 0) && (sdll < mindll)
                    break
                end
            end
        end


        if NAN_CHECK_ACTIVE
            check_nan(myAmica)
        end



        next!(
            p,
            showvalues = [("iter", iter), ("LL", myAmica.LL[iter]), ("lrate", lrate.lrate)],
        )

    end

    if show_timing
        print_timer(to)
    end
    finish!(p)
    if sort_by_variance
        sort_according_to_var!(myAmica, data)
    end
    return myAmica
end


function sort_according_to_var!(
    myAmica::Amica.SingleModelAmica,
    data::AbstractMatrix{T},
) where {T<:Real}
    @warn "It is unclear right now whether this function works correctly, as the calculated variances are all very close to 1 - we might have missed a normalisation step somwhere. Your mileage might vary"
    # Get the variances of the recovered sources
    winv = (mixing(myAmica))
    #meanvar = sum(winv.^2).*sum((data').^2)/((chans*frames)-1); % from Rey Ramirez 8/07
    variances = dropdims(sum(winv .^ 2, dims = 1) .* sum(data, dims = 1) .^ 2, dims = 1)
    #        svar(i,h) = sum( alpha(1:num_mix_used(i,h),i,h) .* (mu(1:num_mix_used(i,h),i,h).^2 + ...
    #    (gamma(3./rho(1:num_mix_used(i,h),i,h))./gamma(1./rho(1:num_mix_used(i,h),i,h)))./sbeta(1:num_mix_used(i,h),i,h).^2) );
    # svar(i,h) = svar(i,h) * norm(A(:,i,h))^2;
    svar = similar(variances)
    N, n, m = myAmica.dims
    A = Matrix(myAmica.A)
    proportions = Matrix(myAmica.proportions)
    location = Matrix(myAmica.location)
    scale = Matrix(myAmica.scale)
    shape = Matrix(myAmica.shape)

    for i = 1:n # go over components
        svar[i] = sum(
            proportions[i, :] .* (
                location[i, :] .^ 2 .+
                (gamma.(3 ./ shape[i, :]) ./ gamma.(1 ./ shape[i, :])) ./ scale[i, :] .^ 2
            ),
        )
        svar[i] *= norm(A[:, i])^2
    end

    #@info "variances" variances, svar
    # from eeglab / rey Ramirez - ommitted constant normalization
    #variances = sum((data * S').^2, dims=1) ./ (size(data, 1) - 1)

    # Get the sorting indices based on variances (descending order)
    sorted_indices = sortperm(svar, rev = true)
    #@info sorted_indices
    # Sort the unmixing matrix and other parameters according to the sorted indices
    # Sort the unmixing matrix and other parameters according to the sorted indices
    myAmica = deepcopy(myAmica)
    myAmica.A = myAmica.A[:, sorted_indices]
    myAmica.proportions = myAmica.proportions[sorted_indices, :]
    myAmica.scale = myAmica.scale[sorted_indices, :]
    myAmica.location = myAmica.location[sorted_indices, :]
    myAmica.shape = myAmica.shape[sorted_indices, :]
    #myAmica.S = myAmica.S[:,sorted_indices]

    return myAmica
end


"""
    recover_sources(data, myAmica)

Recover sources from mixtures `data` using a fitted `SingleModelAmica`.

sources = data * unmixing

[`mixing`](@ref) [`unmixing`](@ref)
"""
recover_sources(data, myAmica::AbstractAmica) = data * unmixing(myAmica)

"""
    unmixing(myAmica::SingleModelAmica)
The unmixing matrix / spatial filter, W*y=s

From sensors to sources.

Sphering is already included.

# See also
[`mixing`](@ref) [`recover_sources`](@ref)
"""
unmixing(x::AbstractAmica) = sphering(x) * inv(Matrix(x.A))'

"""
    mixing(myAmica::SingleModelAmica)
The mixing matrix / spatial map, y = A * s

From sources to sensors.

Inverse sphering is already included.

# See also
[`unmixing`](@ref) [`recover_sources`](@ref)
"""
mixing(x::AbstractAmica) = Matrix(x.A)' * inv(sphering(x))

sphering(myAmica) = Matrix(myAmica.S)
