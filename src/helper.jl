#removes mean from nxN float matrix
function removeMean!(input)
    mn = mean(input, dims=1)
    (_, n) = size(input)
    for i in 1:n
        input[:, i] .= input[:, i] .- mn[i]
    end
    return mn
end

#Returns sphered data x. todo:replace with function from lib
function sphering!(x)
    (N, _) = size(x)
    F = svd(x' * x / N)
    S = F.U * diagm(1 ./ sqrt.(F.S)) * F.U'
    x .= x * S
    return S
end

#Adds means back to model centers
add_means_back!(myAmica::SingleModelAmica, removed_mean) = nothing


function add_means_back!(myAmica::MultiModelAmica, removed_mean)
    (_, _, m) = size(myAmica.models, 1)
    for h in 1:m
        myAmica.models[h].centers = myAmica.models[h].centers + removed_mean #add mean back to model centers
    end
end

function notzero(val::T) where T<:Real
    epsilon = T(1e-16)
    if val < epsilon && val > -epsilon
        # Don't use sign(val) because sign(0) = 0, which gives 0 * epsilon = 0
        ifelse(val >= T(0), epsilon, -epsilon)
    else
        val
    end
end


# copied from specialfunctions but adapted to use Base.Math._evalpoly instead of the macro which won't run on the gpu
function gpuDigamma(z::T) where T<:Real
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
    c = (T(0.08333333333333333),
        T(-0.008333333333333333),
        T(0.003968253968253968),
        T(-0.004166666666666667),
        T(0.007575757575757576),
        T(-0.021092796092796094),
        T(0.08333333333333333),
        T(-0.4432598039215686))
    ψ -= t * Base.Math._evalpoly(t, c)
end


function check_nan(myAmica::SingleModelAmica)
    if any(isnan, myAmica.shape)
        @warn "NaN in myAmica.shape"
    end
    if any(isnan, myAmica.y)
        @warn "NaN in myAmica.y"
    end
    if any(isnan, myAmica.y_rho)
        @warn "NaN in myAmica.y_rho"
    end
    if any(isnan, myAmica.dA)
        @warn "NaN in myAmica.dA"
    end
    if any(isnan, myAmica.Lt)
        @warn "NaN in myAmica.Lt"
    end
    if any(isnan, myAmica.z)
        @warn "NaN in myAmica.z"
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
    if any(isnan, myAmica.shape)
        @warn "NaN in myAmica.shape"
    end
    if any(isnan, myAmica.source_signals)
        @warn "NaN in myAmica.source_signals"
    end
    if any(isnan, myAmica.y)
        @warn "NaN in myAmica.y"
    end
end