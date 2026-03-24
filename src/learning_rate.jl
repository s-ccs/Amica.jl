#Structure for Learning Rate type with initial value, minumum, maximum etc. Used for learning rate and shape lrate
mutable struct LearningRate{T}
    lrate::T
    lrate0::T
    shapelrate::T
    shapelrate0::T
    shape0::T
    lratefact::T
    shapelratefact::T
    min::T
    maxdecs::T
    max_incs::Int
    use_min_dll::Bool
    min_dll::T
    min_nd::T
    numdecs::Int
    numincs::Int
    newtrate::T
    newt_ramp::Int
    minrho::T
    maxrho::T
end

function LearningRate{T}(lrate::T=T(0.1), shapelrate::T=T(0.05);
    shape0::T=T(1.5),
    lratefact::T=T(0.5),
    shapelratefact::T=T(0.1),
    min::T=T(1.0e-12),
    maxdecs::T=T(3),
    max_incs::Int=5,
    use_min_dll::Bool=true,
    min_dll::T=T(1e-9),
    min_nd::T=T(1e-7),
    numdecs::Int=0,
    numincs::Int=0,
    newtrate::T=T(0.5),
    newt_ramp::Int=10,
    minrho::T=T(1.0),
    maxrho::T=T(2.0)
) where {T<:Real}
    LearningRate{T}(lrate, copy(lrate), shapelrate,
        copy(shapelrate), shape0, lratefact,
        shapelratefact, min, maxdecs, max_incs,
        use_min_dll, min_dll, min_nd,
        numdecs, numincs, newtrate, newt_ramp,
        minrho, maxrho
    )
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
