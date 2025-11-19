"""
Main AMICA algorithm

"""

function fit(amicaType::Type{T}, data; m=3, maxiter=500, location=nothing, scale=nothing, A=nothing, kwargs...) where {T<:AbstractAmica}
    amica = T(data; m=m, maxiter=maxiter, location=location, scale=scale, A=A)
    fit!(amica, data; kwargs...)
    return amica
end
function fit!(amica::AbstractAmica, data; kwargs...)
    amica!(amica, data; kwargs...)
end

function amica!(myAmica::AbstractAmica,
    data::AbstractMatrix{T};
    lrate::LearningRate{T}=LearningRate{T}(),
    shapelrate::LearningRate{T}=LearningRate{T}(; lrate=0.05, minimum=1.0, maximum=2.0, init=1.5),
    remove_mean::Bool=true,
    do_sphering::Bool=true,
    show_progress::Bool=true,
    maxiter::Int=myAmica.maxiter,
    do_newton::Bool=true,
    newt_start_iter::Int=50,
    iterwin::Int=10,
    update_shape::Bool=true,
    mindll::T=T(1e-8), kwargs...) where {T<:Real}

    initialize_shape_parameter!(myAmica, shapelrate)

    #Prepares data by removing means and/or sphering
    if remove_mean
        removed_mean = removeMean!(data)
    end

    if do_sphering
        S = sphering!(data)
        myAmica.S = S
        myAmica.LLdetS = logabsdet(S)[1]
    else
        myAmica.S = I
        myAmica.LLdetS = 0
    end

    dLL = zeros(1, maxiter)

    prog = ProgressUnknown("Minimizing"; showspeed=true)

    for iter in 1:maxiter
        iter_time_start = time()

        @timeit to "update_sources" update_sources!(myAmica, data)

        @timeit to "calculate_ldet" calculate_ldet!(myAmica)

        @timeit to "calculate_y" calculate_y!(myAmica)

        @timeit to "update_y_rho" update_y_rho!(myAmica)

        @timeit to "loopiloop" loopiloop!(myAmica) #Updates y and Lt. Todo: Rename

        @timeit to "calculate_DLL" calculate_DLL!(dLL, myAmica, iter)

        if iter > iterwin + 1
            @timeit to "calculate_lrate" calculate_lrate!(dLL, lrate, iter, newt_start_iter, do_newton, iterwin)
            #Calculates average likelihood change over multiple itertions

            #Checks termination criterion
            @timeit to "calculate_sdLL" sdll = calculate_sdLL(dLL, iter, iterwin)
            if (sdll > 0) && (sdll < mindll)
                println("LL increase to low. Stop at iteration ", iter)
                break
            end
        end

        #M-step
        try
            #Updates parameters and mixing matrix
            # [OK]
            @timeit to "update_loop" update_loop!(myAmica, shapelrate, update_shape, iter, do_newton, newt_start_iter, lrate)
        catch e
            #Terminates if NaNs are detected in parameters
            if isa(e, AmicaNaNException)
                println("\nNaN detected. Better stop. Current iteration: ", iter)
                @goto escape_from_NaN
            else
                rethrow()
            end
        end

        @timeit to "reparameterize" reparameterize!(myAmica, data)

        # Calculate iteration time
        iter_time = time() - iter_time_start

        # Formatted output matching Fortran AMICA
        if show_progress
            println(" iter $(lpad(iter, 5)) lrate = $(lpad(string(round(lrate.lrate, digits=10)), 13)) LL = $(lpad(string(round(myAmica.LL[iter], digits=10)), 14))  ($(lpad(string(round(iter_time, digits=2)), 6)) s)")
        end

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