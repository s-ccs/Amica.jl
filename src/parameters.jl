#Normalizes source density location parameter (mu), scale parameter (beta) and model centers
function reparameterize!(myAmica::SingleModelAmica)
    ArrayType = typeof(myAmica.Lt)
    T = eltype(myAmica.A)
    tau = ArrayType(norm.(eachcol(myAmica.A)))

    # Only reparameterize columns where tau > 0 (matching Fortran behavior)
    mask = tau .> zero(T)
    myAmica.A .= ifelse.(push_dimension(mask), myAmica.A ./ tau', myAmica.A)
    myAmica.location .= ifelse.(mask, myAmica.location .* tau, myAmica.location)
    myAmica.scale .= ifelse.(mask, myAmica.scale ./ tau, myAmica.scale)
end

#Reparameterizes the parameters for the active models
function reparameterize!(myAmica::MultiModelAmica, data)
    (N, _, n) = size(myAmica.models[1].source_signals)
    M = size(myAmica.models, 1)

    for h = 1:M
        mu = myAmica.models[h].location
        beta = myAmica.models[h].scale

        if myAmica.normalized_ica_weights[h] == 0
            continue
        end
        for i in 1:n
            tau = norm(myAmica.models[h].A[:, i])
            myAmica.models[h].A[:, i] = myAmica.models[h].A[:, i] / tau
            mu[:, i] = mu[:, i] * tau
            beta[:, i] = beta[:, i] / tau
        end

        if M > 1
            cnew = data * myAmica.ica_weights_per_sample[h, :] / (sum(myAmica.ica_weights_per_sample[h, :])) #todo: check why v not inverted
            for i in 1:n
                Wh = pinv(myAmica.models[h].A[:, :])
                mu[:, i] = mu[:, i] .- Wh[i, :]' * (cnew - myAmica.models[h].centers[:])
            end
            myAmica.models[h].centers = cnew
        end
        myAmica.models[h].location .= mu
        myAmica.models[h].scale .= beta
    end
end

# Sets the initial value for the shape parameter of the GeneralizedGaussians for each Model
function initialize_shape_parameter!(myAmica::SingleModelAmica, lrate::LearningRate)
    myAmica.shape .= lrate.shape0 .* myAmica.shape
end

function initialize_shape_parameter!(myAmica::MultiModelAmica, lrate::LearningRate)
    initialize_shape_parameter!.(myAmica.models, lrate)
end

function update_parameters!(myAmica::MultiModelAmica{T}, lrate::LearningRate, upd_shape::Bool) where {T<:Real}
    update_parameters!.(myAmica.models, lrate, upd_shape)
end


# copied from specialfunctions
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


function fp(myAmica::SingleModelAmica)
    myAmica.y_rho .* sign.(myAmica.y) .* push_dimension(myAmica.shape)
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


#Updates Gaussian mixture parameters. It also returns g, kappa and lamda which are needed to apply the newton method.
@views function update_parameters!(myAmica::SingleModelAmica{T}, lrate::LearningRate, upd_shape::Bool, newton_active::Bool) where {T<:Real}
    N, _, m = size(myAmica.y)

    check_isnan = false

    if check_isnan && any(isnan, myAmica.y)
        @warn "NaN in myAmica.y"
    end
    if check_isnan && any(isnan, myAmica.z)
        @warn "NaN in myAmica.z"
    end
    if check_isnan && any(isnan, myAmica.y_rho)
        @warn "NaN in myAmica.y_rho"
    end

    @timeit to "kernel" begin
        # sum(z)
        sum_z = sum(myAmica.z, dims=1)[1, :, :]
        # sum(z * fp)
        dmu_numer = sum(myAmica.z .* fp(myAmica), dims=1)[1, :, :]
        # sum(z * fp * fp)
        kp = sum(myAmica.z .* fp(myAmica) .^ 2, dims=1)[1, :, :]
        # rho <= 2 ? sum(z * fp / y) : sum(z * fp * fp)
        # the 'notzero' clamping isn't present in fortran but helps keeping values numerically stable for float32
        dmu_denom = sum(ifelse.(push_dimension(myAmica.shape) .<= T(2),
                myAmica.z .* fp(myAmica) ./ notzero.(myAmica.y),
                myAmica.z .* fp(myAmica) .^ 2
            ), dims=1)[1, :, :] .* myAmica.scale

        # sum(z * log(y_rho) * y_rho)
        drho_numer = sum(ifelse.(
                myAmica.y_rho .>= T(1.0e-16), myAmica.z .* log.(myAmica.y_rho) .* myAmica.y_rho,
                T(0.0)
            ), dims=1)[1, :, :]

        # sum(scale * z * fp)
        myAmica.g .= sum(push_dimension(myAmica.scale) .* myAmica.z .* fp(myAmica), dims=3)[:, :, 1]
        # sum(z * (fp * y - 1)^2)
        dlambda_numer = sum(myAmica.z .* (fp(myAmica) .* myAmica.y .- T(1.0)) .^ 2, dims=1)[1, :, :]
        # rho <= 2.0 ? sum(z * fp * y) : 0
        dbeta_denom = sum(ifelse.(push_dimension(myAmica.shape) .<= T(2), myAmica.z .* fp(myAmica) .* myAmica.y, T(0)), dims=1)[1, :, :]
    end

    if check_isnan && any(isnan, sum_z)
        @warn "NaN in sum_z"
    end
    if check_isnan && any(isnan, dmu_numer)
        @warn "NaN in dmu_numer"
    end
    if check_isnan && any(isnan, kp)
        @warn "NaN in kp"
    end
    if check_isnan && any(isnan, dmu_denom)
        @warn "NaN in dmu_denom"
    end
    if check_isnan && any(isnan, drho_numer)
        @warn "NaN in drho_numer"
    end
    if check_isnan && any(isnan, myAmica.g)
        @warn "NaN in myAmica.g"
    end
    if check_isnan && any(isnan, dlambda_numer)
        @warn "NaN in dlambda_numer"
    end
    if check_isnan && any(isnan, dbeta_denom)
        @warn "NaN in dbeta_denom"
    end

    # alpha / proportions
    @timeit to "prop" if m > 1
        myAmica.proportions .= ifelse.(sum_z .>= T(0), sum_z ./ N, T(1) / N)
    end

    # newton parameters
    @timeit to "para" if newton_active
        dkap = @. (kp / (myAmica.proportions * N)) * myAmica.scale^2
        myAmica.newton_kappa .= sum(@. myAmica.proportions * dkap; dims=2)
        myAmica.newton_lambda .= sum(@. myAmica.proportions * (dlambda_numer / sum_z + dkap * myAmica.location^2); dims=2)
    end

    # mu / location
    @timeit to "loc" if m > 1
        myAmica.location .+= dmu_numer ./ dmu_denom
    end

    # sbeta / scale
    @timeit to "scale" begin
        # dbeta_numer / dbeta_denom
        myAmica.scale .*= sqrt.(sum_z ./ dbeta_denom)
    end

    # rho / shape
    @timeit to "shape" if upd_shape
        myAmica.shape .= clamp.(
            myAmica.shape .+ (lrate.shapelrate .* (1 .- (myAmica.shape ./ gpuDigamma.(1 .+ 1 ./ myAmica.shape)) .* drho_numer ./ sum_z)),
            lrate.minrho,
            lrate.maxrho
        )
    end

    if check_isnan && any(isnan, myAmica.proportions)
        @warn "NaN in myAmica.proportions"
    end
    if check_isnan && any(isnan, myAmica.newton_kappa)
        @warn "NaN in myAmica.newton_kappa"
    end
    if check_isnan && any(isnan, myAmica.newton_lambda)
        @warn "NaN in myAmica.newton_lambda"
    end
    if check_isnan && any(isnan, myAmica.location)
        @warn "NaN in myAmica.location"
    end
    if check_isnan && any(isnan, myAmica.scale)
        @warn "NaN in myAmica.scale"
    end
    if check_isnan && any(isnan, myAmica.shape)
        @warn "NaN in myAmica.shape"
    end
end

"updates the unmixed source_signals: myAmica.source_signals = myAmica.A ^ -1 * data"
function update_sources!(myAmica::SingleModelAmica{T}, data::AbstractMatrix{T}) where {T<:Real}
    myAmica.source_signals .= ((myAmica.A |> Array) \ (data' |> Array))' |> typeof(myAmica.source_signals)
end

function update_sources!(myAmica::MultiModelAmica, data)
    n = size(myAmica.models[1].A, 1)
    for h in 1:length(myAmica.models)
        for i in 1:n
            Wh = pinv(myAmica.models[h].A)
            myAmica.models[h].source_signals[i, :] = Wh[i, :]' * data .- Wh[i, :]' * myAmica.models[h].centers
        end
    end
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