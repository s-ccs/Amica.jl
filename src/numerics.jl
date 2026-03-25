"""
    notzero(val::T)::T where T<:Real

Return value or epsilon with preserved sign to avoid division by zero.

Returns the input value if it is not near zero, otherwise returns a small positive or
negative epsilon value matching the sign of the input. This is useful for safe division
operations or logarithm calculations that should not receive exactly zero values.

# Arguments
- `val::T`: A real number to check.

# Returns
- `result::T`: Either `val` if it is not near zero, or ±epsilon ("1e-16) matching the sign of `val`.

# Examples
```julia-repl
julia> notzero(1.0)
1.0

julia> notzero(0.0)
1.0e-16

julia> notzero(-1e-20)
-1.0e-16
```
"""
@views function notzero(val::T) where {T<:Real}
    epsilon = T(1e-16)
    if val < epsilon && val > -epsilon
        # Don't use sign(val) because sign(0) = 0, which gives 0 * epsilon = 0
        ifelse(val >= T(0), epsilon, -epsilon)
    else
        val
    end
end

"Calculate difference in loglikelihood between iterations"
@views function calculate_DLL!(dLL, myAmica::SingleModelAmica, iter)
    if iter > 1
        dLL[iter] = myAmica.LL[iter] - myAmica.LL[iter-1]
    end
end

"add a unit dimension in front to be able to e.g. broadcast a (1000, 12, 3) with a (12, 3) array, transforms a from (12, 3) to (1, 12, 3)"
@views function push_dimension(a::AbstractArray)
    reshape(a, 1, size(a)...)
end


"""
    gpuDigamma(z::T)::T where T<:Real

Compute the digamma function (logarithmic derivative of gamma function), GPU-compatible.

This is an implementation of the digamma function adapted from SpecialFunctions.jl that
uses Base.Math._evalpoly instead of macros, making it GPU-compatible for use with GPU arrays.
The digamma function is the logarithmic derivative of the gamma function, i.e., ψ(z) = d(ln Γ(z))/dz.

# Arguments
- `z::T`: Input value (complex or real).

# Returns
- `result::T`: The digamma value at z.

# Examples
```julia-repl
julia> gpuDigamma(1.0)  # digamma(1) = -0.5772...
-0.5772156649015328

julia> gpuDigamma(2.0)  # digamma(2) = 0.4227...
0.422784335098467
```
"""
@views function gpuDigamma(z::T) where {T<:Real}
    # Based on eq. (12), without looking at the accompanying source
    # code, of: K. S. Kölbig, "Programs for computing the logarithm of
    # the gamma function, and the digamma function, for complex
    # argument," Computer Phys. Commun.  vol. 4, pp. 221–226 (1972).
    x = real(z)
    if x <= 0 # reflection formula
        ψ = -T(π) / tanpi(z)
        z = 1 - z
        x = real(z)
    else
        ψ = zero(z)
    end
    X = 8
    if x < X
        # shift using recurrence formula
        n = X - unsafe_trunc(Int, x)
        for ν = 1:n-1
            ψ -= inv(z + ν)
        end
        ψ -= inv(z)
        z += n
    end
    t = inv(z)
    ψ += log(z) - T(0.5) * t
    t *= t # 1/z^2
    # the coefficients here are Float64(bernoulli[2:9] .// (2*(1:8)))
    c = (
        T(0.08333333333333333),
        T(-0.008333333333333333),
        T(0.003968253968253968),
        T(-0.004166666666666667),
        T(0.007575757575757576),
        T(-0.021092796092796094),
        T(0.08333333333333333),
        T(-0.4432598039215686),
    )
    ψ -= t * Base.Math._evalpoly(t, c)
end


"""
    check_nan(myAmica::SingleModelAmica)

Check for NaN values in AMICA model parameters and warn if found.

Detects NaN values in all major model parameter arrays and issues warnings for debugging.
Useful for diagnosing convergence issues or numerical problems during model fitting.

# Arguments
- `myAmica::SingleModelAmica`: The AMICA model to check.

# Examples
```julia-repl
julia> check_nan(myAmica)  # Issues warnings if NaN values are found
```
"""
function check_nan(myAmica::SingleModelAmica)
    if any(isnan, myAmica.shape)
        @warn "NaN in myAmica.shape"
    end
    if any(isnan, myAmica.dA)
        @warn "NaN in myAmica.dA"
    end
    if any(isnan, myAmica.Lt)
        @warn "NaN in myAmica.Lt"
    end
    if any(isnan, myAmica.proportions)
        @warn "NaN in myAmica.proportions"
    end
    if any(isnan, myAmica.newton_kappa)
        @warn "NaN in myAmica.newton_kappa"
    end
    if any(isnan, myAmica.newton_lambda)
        @warn "NaN in myAmica.newton_lambda"
    end
    if any(isnan, myAmica.location)
        @warn "NaN in myAmica.location"
    end
    if any(isnan, myAmica.scale)
        @warn "NaN in myAmica.scale"
    end
end
