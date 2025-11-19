#Normalizes source density location parameter (mu), scale parameter (beta) and model centers
function reparameterize!(myAmica::SingleModelAmica)
    tau = norm.(eachcol(myAmica.A))

    myAmica.A ./= tau'
    myAmica.location .*= tau
    myAmica.scale ./= tau
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
    myAmica.shape = lrate.shape0 .* myAmica.shape
end

function initialize_shape_parameter!(myAmica::MultiModelAmica, lrate::LearningRate)
    initialize_shape_parameter!.(myAmica.models, lrate)
end

function update_parameters!(myAmica::MultiModelAmica{T}, lrate::LearningRate, upd_shape::Bool) where {T<:Real}
    update_parameters!.(myAmica.models, lrate, upd_shape)
end

#Updates Gaussian mixture parameters. It also returns g, kappa and lamda which are needed to apply the newton method.
@views function update_parameters!(myAmica::SingleModelAmica{T}, lrate::LearningRate, upd_shape::Bool, newton_active::Bool) where {T<:Real}
    N, n, m = size(myAmica.y)

    myAmica.g .= zero(T)
    myAmica.newton_kappa .= zero(T)
    myAmica.newton_lambda .= zero(T)

    for j in 1:m, i in 1:n
        sum_zfp = zero(T)
        sum_z = zero(T)
        dm = zero(T)
        dbeta_denom = zero(T)
        drho_numer = zero(T)
        kp = zero(T)
        dlambda_numer = zero(T)

        for k in 1:N
            fp = myAmica.y_rho[k, i, j] * sign(myAmica.y[k, i, j]) * myAmica.shape[i, j]

            zfp = myAmica.z[k, i, j] * fp
            sum_zfp += zfp
            sum_z += myAmica.z[k, i, j]

            # kp = sum(z * fp * fp) for use in location update and Newton method
            kp += zfp * fp
            dm += zfp / myAmica.y[k, i, j]

            myAmica.g[k, i] += myAmica.scale[i, j] * zfp

            if myAmica.shape[i, j] <= 2
                dbeta_denom += zfp * myAmica.y[k, i, j]
            else
                dbeta_denom += myAmica.z[k, i, j] * myAmica.y_rho[k, i, j]
            end
            drho_numer += myAmica.z[k, i, j] * log(myAmica.y_rho[k, i, j]) * myAmica.y_rho[k, i, j]

            if newton_active
                dlambda_numer += myAmica.z[k, i, j] * (fp * myAmica.y[k, i, j] - 1.0)^2
            end
        end

        if dm <= 0
            dm = one(T)
        end

        # update proportions
        if m > 1
            if sum_z >= 0
                myAmica.proportions[i, j] = sum_z ./ N
            else
                myAmica.proportions[i, j] = 1 ./ N
            end
        end



        # newton parameters
        if newton_active
            dkap = (kp / (myAmica.proportions[i, j] * N)) * myAmica.scale[i, j]^2
            myAmica.newton_kappa[i] += myAmica.proportions[i, j] * dkap
            myAmica.newton_lambda[i] += myAmica.proportions[i, j] * (dlambda_numer / sum_z + dkap * myAmica.location[i, j]^2)
        end

        # update location
        if m > 1
            if myAmica.shape[i, j] <= 2
                myAmica.location[i, j] += (1 / myAmica.scale[i, j]) * sum_zfp / dm
            elseif kp > 0
                # Fortran: mu += dmu_numer / dmu_denom = sum(z*fp) / (scale * sum(z*fp*fp))
                # Now kp = sum(z*fp*fp), so:
                myAmica.location[i, j] += sum_zfp / (myAmica.scale[i, j] * kp)
            end
        end

        # update scale
        if dbeta_denom > 0
            myAmica.scale[i, j] *= sqrt(sum_z / dbeta_denom)
        end

        # update shape
        if upd_shape && sum_z > 0
            dr2 = 1 - (myAmica.shape[i, j] / digamma(1 + 1 / myAmica.shape[i, j])) * drho_numer / sum_z
            if !isnan(dr2)
                myAmica.shape[i, j] += lrate.shapelrate * dr2
                myAmica.shape[i, j] = clamp(myAmica.shape[i, j], lrate.minrho, lrate.maxrho)
            end
        end
    end

    if any(isnan, myAmica.source_signals) || any(isnan, myAmica.g) || any(isnan, myAmica.proportions)
        throw(AmicaNaNException())
    end
end



"updates the unmixed source_signals: myAmica.source_signals = myAmica.A ^ -1 * data"
function update_sources!(myAmica::SingleModelAmica{T}, data::AbstractMatrix{T}) where {T<:Real}
    myAmica.source_signals = ((myAmica.A |> Array) \ (data' |> Array))' |> typeof(myAmica.source_signals)
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
end