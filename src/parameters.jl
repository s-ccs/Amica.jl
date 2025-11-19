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
function initialize_shape_parameter!(myAmica::SingleModelAmica, shapelrate::LearningRate)
    myAmica.shape = shapelrate.init .* myAmica.shape
end

function initialize_shape_parameter!(myAmica::MultiModelAmica, shapelrate::LearningRate)
    initialize_shape_parameter!.(myAmica.models)
end

function update_parameters!(myAmica::MultiModelAmica{T}, shapelrate::LearningRate, upd_shape::Bool) where {T<:Real}
    update_parameters!.(myAmica.models, shapelrate, upd_shape)
end

#Updates Gaussian mixture parameters. It also returns g, kappa and lamda which are needed to apply the newton method.
@views function update_parameters!(myAmica::SingleModelAmica{T}, shapelrate::LearningRate, upd_shape::Bool) where {T<:Real}
    N, n, m = size(myAmica.y)
    myAmica.g .= zero(T)

    for j in 1:m, i in 1:n
        sum_zfp = zero(T)
        sum_z = zero(T)
        dm = zero(T)
        dbeta_denom = zero(T)
        drho_numer = zero(T)

        for k in 1:N
            myAmica.fp[k, i, j] = myAmica.y_rho[k, i, j] * sign(myAmica.y[k, i, j]) * myAmica.shape[i, j]

            zfp = myAmica.z[k, i, j] * myAmica.fp[k, i, j]
            sum_zfp += zfp
            sum_z += myAmica.z[k, i, j]

            myAmica.kp[i, j] += zfp * myAmica.fp[k, i, j] * myAmica.scale[i, j]
            dm += zfp / myAmica.y[k, i, j]

            myAmica.g[k, i] += myAmica.scale[i, j] * zfp

            if myAmica.shape[i, j] <= 2
                dbeta_denom += zfp * myAmica.y[k, i, j]
            else
                dbeta_denom += myAmica.z[k, i, j] * myAmica.y_rho[k, i, j]
            end
            drho_numer += myAmica.z[k, i, j] * log(myAmica.y_rho[k, i, j]) * myAmica.y_rho[k, i, j]
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

        # update location
        if m > 1
            if myAmica.shape[i, j] <= 2
                myAmica.location[i, j] += (1 / myAmica.scale[i, j]) * sum_zfp / dm
            elseif myAmica.kp[i, j] > 0
                myAmica.location[i, j] += (1 / myAmica.scale[i, j]) * sum_zfp / myAmica.kp[i, j]
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
                myAmica.shape[i, j] += shapelrate.lrate * dr2
                myAmica.shape[i, j] = clamp(myAmica.shape[i, j], shapelrate.minimum, shapelrate.maximum)
            end
        end
    end

    if any(isnan, myAmica.source_signals) || any(isnan, myAmica.g) || any(isnan, myAmica.proportions)
        throw(AmicaNaNException())
    end
end



"updates the unmixed source_signals: myAmica.source_signals = myAmica.A ^ -1 * data"
function update_sources!(myAmica::SingleModelAmica{T}, data::AbstractMatrix{T}) where {T<:Real}
    myAmica.source_signals .= (myAmica.A \ data')'
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
function calculate_lrate!(dLL, lrateType::LearningRate, iter, newt_start_iter, do_newton, iterwin)

    lratefact, lnatrate, lratemax, = lrateType.decreaseFactor, lrateType.natural_rate, lrateType.maximum
    lrate = lrateType.lrate
    sdll = sum(dLL[iter-iterwin+1:iter]) / iterwin

    if sdll < 0
        println("Likelihood decreasing!")
        lrate = lrate * lratefact
    else
        if (iter > newt_start_iter) && do_newton == 1
            lrate = min(lratemax, lrate + min(0.1, lrate))
        else
            lrate = min(lnatrate, lrate + min(0.1, lrate))
        end
    end
    lrateType.lrate = lrate
end