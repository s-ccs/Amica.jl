"""
Main AMICA algorithm
"""

function fit(amicaType::Type{AmicaKind}, data::Array{T,2}; m=3, maxiter=500, location=nothing, scale=nothing, A=nothing, ArrayType::Type{<:DenseArray}=Array, kwargs...) where {AmicaKind<:AbstractAmica,T<:Real}
    (N, n) = size(data)
    amica = AmicaKind(T, m=m, ncomps=n, nsamples=N, location=location, scale=scale, A=A, ArrayType=ArrayType)
    fit!(amica, data; maxiter=maxiter, kwargs...)
    return amica
end
function fit!(amica::AbstractAmica, data; kwargs...)
    amica!(amica, data; kwargs...)
end

function amica!(myAmica::AbstractAmica,
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
    mindll::T=T(1e-8)) where {T<:Real}

    amica_start = time()

    # Check that data dimensions match the model
    if size(data) != size(myAmica.source_signals)
        error("Data dimension mismatch: data has size $(size(data)) but model expects $(size(myAmica.source_signals))")
    end

    @timeit to "initialize_shape_parameter!" initialize_shape_parameter!(myAmica, lrate)

    #Prepares data by removing means and/or sphering
    @timeit to "removeMean" if remove_mean
        removed_mean = removeMean!(data)
    end

    @timeit to "sphering" if do_sphering
        S = sphering!(data)
        myAmica.S = S
        myAmica.LLdetS = logabsdet(S |> Array)[1]
    else
        myAmica.S = I
        myAmica.LLdetS = 0
    end

    dLL = zeros(1, maxiter)
    iter_times = Float64[]

    # TODO make previous operations run on gpu as well
    # move to gpu

    @timeit to "movetogpu" if typeof(myAmica.A) != typeof(data)
        data = data |> typeof(myAmica.A)
    end

    if show_progress
        preparation_time = time() - amica_start
        println("\nPreparation completed, starting main loop ($(round(preparation_time, digits=3)) s)")
    end
    niter = 0
    loop_start = time()

    for iter in 1:maxiter
        niter += 1
        iter_time_start = time()

        @timeit to "update_sources" begin
            update_sources!(myAmica, data)
        end

        @timeit to "calculate_y" begin
            calculate_y!(myAmica)
        end

        @timeit to "update_y_rho" begin
            calculate_y_rho!(myAmica)
        end

        @timeit to "calculate_u_and_Lt" begin
            calculate_u_and_Lt!(myAmica)
        end

        @timeit to "calculate_DLL" begin
            calculate_DLL!(dLL, myAmica, iter)
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

        #M-step
        try
            #Updates parameters and mixing matrix
            @timeit to "update_parameters" begin
                update_parameters!(myAmica, lrate, update_shape, do_newton && iter >= newt_start_iter)
            end
            @timeit to "update_mixing" begin
                update_mixing!(myAmica, iter, do_newton, newt_start_iter, lrate)
            end
        catch e
            #Terminates if NaNs are detected in parameters
            if isa(e, AmicaNaNException)
                println("\nNaN detected. Better stop. Current iteration: ", iter)
                @goto escape_from_NaN
            else
                rethrow()
            end
        end

        @timeit to "reparameterize" begin
            reparameterize!(myAmica)
        end

        # TODO remove
        if iter == 1 && maxiter > 1
            reset_timer!(to)
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
    #If parameters contain NaNs, the algorithm skips the A update and terminates by jumping here
    @label escape_from_NaN

    if show_progress
        print_timer(to, sortby=:allocations)
        # Log average iteration time
        avg_iter_time = (time() - loop_start) / niter
        println("\nAverage iteration time: $(round(avg_iter_time, digits=3)) s (over $(niter) iterations)")
    end

    #If means were removed, they are added back
    if remove_mean
        add_means_back!(myAmica, removed_mean)
    end
    @debug myAmica.LL
    return myAmica
end