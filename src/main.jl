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
    shapelrate::LearningRate{T}=LearningRate{T}(; lrate=0.1, minimum=0.5, maximum=5, init=1.5),
    remove_mean::Bool=true,
    do_sphering::Bool=true,
    show_progress::Bool=true,
    maxiter::Int=myAmica.maxiter,
    do_newton::Bool=true,
    newt_start_iter::Int=25,
    iterwin::Int=10,
    update_shape::Bool=true,
    mindll::T=1e-8, kwargs...) where {T<:Real}

    initialize_shape_parameter!(myAmica, shapelrate)

    (n, N) = size(data)
    m = myAmica.m

    println("m[j]: $(m), n[i]: $(n), N: $(N)")

    #Prepares data by removing means and/or sphering
    if remove_mean
        removed_mean = removeMean!(data)
    end
    if do_sphering
        S = sphering!(data)
        myAmica.S = S
        LLdetS = logabsdet(S)[1]
    else
        myAmica.S = I
        LLdetS = 0
    end

    dLL = zeros(1, maxiter)

    prog = ProgressUnknown("Minimizing"; showspeed=true)

    for iter in 1:maxiter
        gg = myAmica.learnedParameters
        @debug :scale, gg.scale[2], :location, gg.location[2], :proportions, gg.proportions[2], :shape, gg.shape[2]
        @debug println("")
        #E-step
        #@show typeof(myAmica.A),typeof(data)
        @debug :A myAmica.A[1:2, 1:2]

        @debug :source_signals myAmica.source_signals[1], myAmica.source_signals[end]
        update_sources!(myAmica, data)
        @debug :source_signals myAmica.source_signals[1], myAmica.source_signals[end]
        calculate_ldet!(myAmica)
        initialize_Lt!(myAmica)
        myAmica.Lt .+= LLdetS

        calculate_y!(myAmica)

        @debug :y, myAmica.y[1, 1:3, 1]
        # pre-calculate abs(y)^rho
        myAmica.y_rho .= abs.(myAmica.y)
        for i in 1:n
            for j in 1:m
                @views _y_rho = myAmica.y_rho[j, i, :]
                optimized_pow!(_y_rho, _y_rho, myAmica.learnedParameters.shape[j, i])
            end
        end



        loopiloop!(myAmica) #Updates y and Lt. Todo: Rename


        calculate_LL!(myAmica)


        @debug (:LL, myAmica.LL)
        #Calculate difference in loglikelihood between iterations
        if iter > 1
            dLL[iter] = myAmica.LL[iter] - myAmica.LL[iter-1]
        end
        if iter > iterwin + 1
            calculate_lrate!(dLL, lrate, iter, newt_start_iter, do_newton, iterwin)
            #Calculates average likelihood change over multiple itertions
            sdll = sum(dLL[iter-iterwin+1:iter]) / iterwin
            #Checks termination criterion
            if (sdll > 0) && (sdll < mindll)
                println("LL increase to low. Stop at iteration ", iter)
                break
            end
        end

        #M-step
        try
            #Updates parameters and mixing matrix
            update_loop!(myAmica, shapelrate, update_shape, iter, do_newton, newt_start_iter, lrate)
        catch e
            #Terminates if NaNs are detected in parameters
            if isa(e, AmicaNaNException)
                println("\nNaN detected. Better stop. Current iteration: ", iter)
                @goto escape_from_NaN
            else
                rethrow()
            end
        end
        @debug iter, :A myAmica.A[1:2, 1:2]

        reparameterize!(myAmica, data)

        @debug iter, :A myAmica.A[1:2, 1:2]
        #Shows current progress
        show_progress && ProgressMeter.next!(prog; showvalues=[(:LL, myAmica.LL[iter])])

    end
    #If parameters contain NaNs, the algorithm skips the A update and terminates by jumping here
    @label escape_from_NaN


    #If means were removed, they are added back
    if remove_mean
        add_means_back!(myAmica, removed_mean)
    end
    @debug myAmica.LL
    return myAmica
end