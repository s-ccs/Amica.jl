#Normalizes source density location parameter (mu), scale parameter (beta) and model centers
function reparameterize!(myAmica::SingleModelAmica{T}) where T<:Real
    (N, n, m) = size(myAmica.z)

    # Calculate norm of column k: Anrmk = sqrt(sum(A(:,k)*A(:,k)))
    tau = sqrt.(sum(myAmica.A .^ 2, dims=1))[1, :]

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



#Updates Gaussian mixture parameters. It also returns g, kappa and lamda which are needed to apply the newton method.
@views function update_parameters!(myAmica::SingleModelAmica{T}, lrate::LearningRate, upd_shape::Bool, newton_active::Bool) where {T<:Real}
    N, _, m = size(myAmica.y)


    @timeit to "kernel" begin
        backend = KernelAbstractions.get_backend(myAmica.z)

        # fp = y_rho * sign(y) * shape
        @timeit to "fp" begin
            fp = myAmica.y_rho .* sign.(myAmica.y) .* push_dimension(myAmica.shape)
        end

        # zfp = z * fp
        @timeit to "zfp" begin
            zfp = myAmica.z .* fp
        end

        # sum(z)
        @timeit to "sum_z" begin
            sum_z = sum(myAmica.z, dims=1)[1, :, :]
        end

        # sum(z * fp)
        @timeit to "dmu_numer" begin
            dmu_numer = sum(zfp, dims=1)[1, :, :]
        end

        # sum(z * fp * fp)
        @timeit to "kp" begin
            myAmica.scratch .= zfp .* fp
            kp = sum(myAmica.scratch, dims=1)[1, :, :]
        end

        # rho <= 2 ? sum(z * fp / y) : sum(z * fp * fp)
        @timeit to "dmu_denom" begin
            myAmica.scratch .= zfp ./ notzero.(myAmica.y)
            dmu_denom = ifelse.(myAmica.shape .<= T(2), sum(myAmica.scratch, dims=1)[1, :, :], kp) .* myAmica.scale
        end

        # sum(z * log(y_rho) * y_rho)
        @timeit to "drho_numer" begin
            myAmica.scratch = myAmica.y_rho .* abs.(myAmica.y)
            myAmica.scratch .= ifelse.(
                myAmica.scratch .>= T(1.0e-16), myAmica.z .* log.(myAmica.scratch) .* myAmica.scratch,
                T(0.0)
            )
            drho_numer = sum(myAmica.scratch, dims=1)[1, :, :]
        end

        # sum(scale * z * fp)
        @timeit to "g" begin
            myAmica.scratch .= push_dimension(myAmica.scale) .* zfp
            myAmica.g .= sum(myAmica.scratch, dims=3)[:, :, 1]
        end

        # sum(z * (fp * y - 1)^2)
        @timeit to "dlambda_numer" begin
            myAmica.scratch .= myAmica.z .* (fp .* myAmica.y .- T(1.0)) .^ 2
            dlambda_numer = sum(myAmica.scratch, dims=1)[1, :, :]
        end

        # rho <= 2.0 ? sum(z * fp * y) : 0
        @timeit to "dbeta_denom" begin
            myAmica.scratch .= ifelse.(push_dimension(myAmica.shape) .<= T(2), zfp .* myAmica.y, T(0))
            dbeta_denom = sum(myAmica.scratch, dims=1)[1, :, :]
        end
    end

    if NAN_CHECK_ACTIVE && any(isnan, sum_z)
        @warn "NaN in sum_z"
    end
    if NAN_CHECK_ACTIVE && any(isnan, dmu_numer)
        @warn "NaN in dmu_numer"
    end
    if NAN_CHECK_ACTIVE && any(isnan, kp)
        @warn "NaN in kp"
    end
    if NAN_CHECK_ACTIVE && any(isnan, dmu_denom)
        @warn "NaN in dmu_denom"
    end
    if NAN_CHECK_ACTIVE && any(isnan, drho_numer)
        @warn "NaN in drho_numer"
    end
    if NAN_CHECK_ACTIVE && any(isnan, myAmica.g)
        @warn "NaN in myAmica.g"
    end
    if NAN_CHECK_ACTIVE && any(isnan, dlambda_numer)
        @warn "NaN in dlambda_numer"
    end
    if NAN_CHECK_ACTIVE && any(isnan, dbeta_denom)
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

    if NAN_CHECK_ACTIVE && any(isnan, myAmica.proportions)
        @warn "NaN in myAmica.proportions"
    end
    if NAN_CHECK_ACTIVE && any(isnan, myAmica.newton_kappa)
        @warn "NaN in myAmica.newton_kappa"
    end
    if NAN_CHECK_ACTIVE && any(isnan, myAmica.newton_lambda)
        @warn "NaN in myAmica.newton_lambda"
    end
    if NAN_CHECK_ACTIVE && any(isnan, myAmica.location)
        @warn "NaN in myAmica.location"
    end
    if NAN_CHECK_ACTIVE && any(isnan, myAmica.scale)
        @warn "NaN in myAmica.scale"
    end
    if NAN_CHECK_ACTIVE && any(isnan, myAmica.shape)
        @warn "NaN in myAmica.shape"
    end
end

"updates the unmixed source_signals: myAmica.source_signals = myAmica.A ^ -1 * data"
function update_sources!(myAmica::SingleModelAmica{T}, data::AbstractMatrix{T}) where {T<:Real}
    W = inv(myAmica.A)
    myAmica.source_signals .= data * W'

    if NAN_CHECK_ACTIVE && any(isnan, myAmica.source_signals)
        @warn "NaN in myAmica.source_signals"
    end

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