"""
Main AMICA algorithm
"""

function fit(amicaType::Type{T}, data; m=3, maxiter=500, location=nothing, scale=nothing, A=nothing, kwargs...) where {T<:AbstractAmica}
    (N, n) = size(data)
    amica = T(m=m, ncomps=n, nsamples=N, location=location, scale=scale, A=A)
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

    # Check that data dimensions match the model
    if size(data) != size(myAmica.source_signals)
        error("Data dimension mismatch: data has size $(size(data)) but model expects $(size(myAmica.source_signals))")
    end

    initialize_shape_parameter!(myAmica, lrate)

    #Prepares data by removing means and/or sphering
    if remove_mean
        removed_mean = removeMean!(data)
    end

    if do_sphering
        S = sphering!(data)
        myAmica.S = S
        myAmica.LLdetS = logabsdet(S |> Array)[1]
    else
        myAmica.S = I
        myAmica.LLdetS = 0
    end

    dLL = zeros(1, maxiter)


    backend = KernelAbstractions.get_backend(myAmica.z)

    # TODO make previous operations run on gpu as well
    # move to gpu
    data = data |> typeof(myAmica.A)

    for iter in 1:maxiter
        iter_time_start = time()

        @timeit to "update_sources" update_sources!(myAmica, data)

        @timeit to "calculate_y" calculate_y!(myAmica)

        @timeit to "calculate_u_and_Lt" calculate_u_and_Lt!(myAmica)

        @timeit to "calculate_DLL" calculate_DLL!(dLL, myAmica, iter)

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
            @timeit to "update_parameters" update_parameters!(myAmica, lrate, update_shape, do_newton && iter >= newt_start_iter)
            @timeit to "newton_method" newton_method!(myAmica, iter, do_newton, newt_start_iter, lrate)
        catch e
            #Terminates if NaNs are detected in parameters
            if isa(e, AmicaNaNException)
                println("\nNaN detected. Better stop. Current iteration: ", iter)
                @goto escape_from_NaN
            else
                rethrow()
            end
        end

        @timeit to "reparameterize" reparameterize!(myAmica)

        # Calculate iteration time
        iter_time = time() - iter_time_start

        # Formatted output matching Fortran AMICA
        if show_progress
            println(" iter $(lpad(iter, 5)) lrate = $(lpad(string(round(lrate.lrate, digits=10)), 13)) LL = $(lpad(string(round(myAmica.LL[iter], digits=10)), 14))  ($(lpad(string(round(iter_time, digits=2)), 6)) s)")
        end

        # TODO remove
        if iter == 1 && maxiter > 1
            reset_timer!(to)
        end

        # TODO is this required?
        KernelAbstractions.synchronize(backend)


    end
    #If parameters contain NaNs, the algorithm skips the A update and terminates by jumping here
    @label escape_from_NaN

    show(to)

    #If means were removed, they are added back
    if remove_mean
        add_means_back!(myAmica, removed_mean)
    end
    @debug myAmica.LL
    return myAmica
end