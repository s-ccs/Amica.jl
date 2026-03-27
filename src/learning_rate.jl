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

"""
    LearningRate{T}(lrate::T=0.1, shapelrate::T=0.05; shape0::T=1.5, lratefact::T=0.5,
                    shapelratefact::T=0.1, min::T=1.0e-12, maxdecs::T=3,
                    max_incs::Int=5, use_min_dll::Bool=true, min_dll::T=1e-9,
                    min_nd::T=1e-7, numdecs::Int=0, numincs::Int=0, newtrate::T=0.5,
                    newt_ramp::Int=10, minrho::T=1.0, maxrho::T=2.0)

Create and initialize a learning rate configuration object.

Constructs a `LearningRate` structure with default or user-specified parameters for
controlling the learning rates and adaptation schedule during AMICA optimization.

# Arguments
- `lrate::T = 0.1`: Initial learning rate for mixing matrix.
- `shapelrate::T = 0.05`: Initial learning rate for shape parameters.

# Keyword Arguments
- `shape0::T = 1.5`: Initial scaling factor for shape parameters.
- `lratefact::T = 0.5`: Multiplication factor when decreasing learning rate.
- `shapelratefact::T = 0.1`: Multiplication factor for shape learning rate decrease.
- `min::T = 1.0e-12`: Minimum learning rate threshold.
- `maxdecs::T = 3`: Maximum allowed learning rate decreases.
- `max_incs::Int = 5`: Maximum consecutive learning rate increases.
- `use_min_dll::Bool = true`: Whether to use minimum change in log-likelihood criterion.
- `min_dll::T = 1e-9`: Minimum required log-likelihood change per iteration.
- `min_nd::T = 1e-7`: Minimum numerical deviation threshold.
- `numdecs::Int = 0`: Initial count of decreases (usually 0).
- `numincs::Int = 0`: Initial count of increases (usually 0).
- `newtrate::T = 0.5`: Learning rate for Newton's method.
- `newt_ramp::Int = 10`: Iterations to ramp up Newton learning rate.
- `minrho::T = 1.0`: Minimum shape parameter value.
- `maxrho::T = 2.0`: Maximum shape parameter value.

# Returns
- `LearningRate{T}`: Initialized learning rate configuration object.

# Examples
```julia-repl
julia> lrate = LearningRate{Float64}(lrate=0.1, shape0=1.5)
julia> lrate.lrate
0.1

julia> lrate = LearningRate{Float32}()  # Use all defaults
```
"""
function LearningRate{T}(
    lrate::T = T(0.1),
    shapelrate::T = T(0.05);
    shape0::T = T(1.5),
    lratefact::T = T(0.5),
    shapelratefact::T = T(0.1),
    min::T = T(1.0e-12),
    maxdecs::T = T(3),
    max_incs::Int = 5,
    use_min_dll::Bool = true,
    min_dll::T = T(1e-9),
    min_nd::T = T(1e-7),
    numdecs::Int = 0,
    numincs::Int = 0,
    newtrate::T = T(0.5),
    newt_ramp::Int = 10,
    minrho::T = T(1.0),
    maxrho::T = T(2.0),
) where {T<:Real}
    LearningRate{T}(
        lrate,
        copy(lrate),
        shapelrate,
        copy(shapelrate),
        shape0,
        lratefact,
        shapelratefact,
        min,
        maxdecs,
        max_incs,
        use_min_dll,
        min_dll,
        min_nd,
        numdecs,
        numincs,
        newtrate,
        newt_ramp,
        minrho,
        maxrho,
    )
end

"""
    calculate_lrate!(myAmica::SingleModelAmica{T}, iter::Int, newt_start_iter::Int,
                     do_newton::Bool, lrate::LearningRate{T})::Bool where T<:Real

Adjust learning rates based on log-likelihood improvement.

Monitors the log-likelihood change between iterations and adaptively adjusts learning rates.
If likelihood decreases, the learning rates are multiplied by a factor less than 1 to slow
convergence. If the learning rate falls below the minimum threshold, returns `true` indicating
optimization should stop.

# Arguments
- `myAmica::SingleModelAmica{T}`: The AMICA model containing log-likelihood history (modified as needed).
- `iter::Int`: Current iteration number.
- `newt_start_iter::Int`: Iteration at which Newton's method starts.
- `do_newton::Bool`: Whether Newton's method is enabled.
- `lrate::LearningRate{T}`: Learning rate configuration and state (modified in-place).

# Returns
- `should_stop::Bool`: Returns `true` if learning rate has decreased to minimum and optimization should stop,
  `false` otherwise.

# Examples
```julia-repl
julia> should_stop = calculate_lrate!(myAmica, 50, 50, true, lrate)

```

# See also
[`update_mixing!`](@ref), [`LearningRate`](@ref)
"""
function calculate_lrate!(
    myAmica::SingleModelAmica{T},
    iter::Int,
    newt_start_iter::Int,
    do_newton::Bool,
    lrate::LearningRate{T},
) where {T<:Real}
    # Check if likelihood is decreasing
    if myAmica.LL[iter] < myAmica.LL[iter-1]
        @info ("Likelihood decreasing!")

        # missing condition: .or. (ndtmpsum .le. min_nd)
        if lrate.lrate <= lrate.min
            @info("minimum change threshold met, exiting loop ...")
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
                    @info("Reducing maximum Newton lrate")
                    lrate.newtrate *= lrate.lratefact
                end
                lrate.numdecs = 0
            end
        end
    end

    false
end
