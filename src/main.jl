"""
Main AMICA algorithm
"""

# TimerOutputs.enable_debug_timings(Amica)

function fit(amicaType::Type{AmicaKind}, data::Array{T,2}; m=3, maxiter=500, location=nothing, scale=nothing, A=nothing, ArrayType::Type{<:DenseArray}=Array, block_size=10_000, num_threads=1, kwargs...) where {AmicaKind<:AbstractAmica,T<:Real}
    (N, n) = size(data)
    amica = AmicaKind(T, m=m, ncomps=n, nsamples=N, location=location, scale=scale, A=A, ArrayType=ArrayType, block_size=block_size, num_threads=num_threads)
    fit!(amica, data; maxiter=maxiter, kwargs...)
    return amica
end
function fit!(amica::AbstractAmica, data; kwargs...)
    amica!(amica, data; kwargs...)
end

function materialize_data_like(target::AbstractMatrix{T}, data::AbstractMatrix{T}) where {T<:Real}
    if data isa typeof(target)
        return copy(data)
    end

    host_data = Matrix{T}(undef, size(data)...)
    copyto!(host_data, data)

    out = similar(target, size(data))
    copyto!(out, host_data)
    return out
end

@views function amica!(myAmica::AbstractAmica,
    data::AbstractMatrix{T};
    lrate::LearningRate{T}=LearningRate{T}(),
    remove_mean::Bool=true,
    do_sphering::Bool=true,
    show_progress::Bool=true,
    maxiter::Int=50,
    do_newton::Bool=true,
    newt_start_iter::Int=50,
    iterwin::Int=10,
    update_shape::Bool=true,
    mindll::T=T(1e-8),
    dump_dir::Union{Nothing,String}=nothing,
    show_timing=false) where {T<:Real}

    amica_start = time()

    @timeit_debug to "initialize_shape_parameter!" initialize_shape_parameter!(myAmica, lrate)
    myAmica.no_newton = false

    #Prepares data by removing means and/or sphering
    if remove_mean
        @timeit_debug to "removeMean" removed_mean = removeMean!(data)
    end

    @timeit_debug to "sphering" if do_sphering
        S = sphering!(data)
        myAmica.S = S
        myAmica.LLdetS = logabsdet(S |> Array)[1]
    else
        myAmica.S = I
        myAmica.LLdetS = 0
    end

    dLL = zeros(1, maxiter)

    @timeit_debug to "materialize_data" data = materialize_data_like(myAmica.A, data)

    if show_progress
        preparation_time = time() - amica_start
        println("\nPreparation completed, starting main loop ($(round(preparation_time, digits=3)) s)")
    end
    niter = 0
    loop_start = time()

    for iter in 1:maxiter
        niter += 1
        iter_time_start = time()

        @timeit_debug to "update_parameters" begin
            update_parameters!(myAmica, data, lrate, update_shape, do_newton && iter >= newt_start_iter; dump_dir)
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
                println("Got NaN! Exiting ...")
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

        # Calculate iteration time
        iter_time = time() - iter_time_start

        # Formatted output matching Fortran AMICA
        if show_progress
            println(" iter $(lpad(iter, 5)) lrate = $(lpad(string(round(lrate.lrate, digits=10)), 13)) LL = $(lpad(string(round(myAmica.LL[iter], digits=10)), 14))  ($(lpad(string(round(iter_time, digits=2)), 6)) s)")
        end
    end

    if show_timing
        print_timer(to)
    end
    if show_progress
        # Log average iteration time
        avg_iter_time = (time() - loop_start) / niter
        println("\nAverage iteration time: $(round(avg_iter_time, digits=3)) s (over $(niter) iterations)")
    end

    return myAmica
end
