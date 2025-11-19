#Normalizes source density location parameter (mu), scale parameter (beta) and model centers
function reparameterize!(myAmica::SingleModelAmica)
    tau = norm.(eachcol(myAmica.A))
    myAmica.A = myAmica.A ./ tau'
    myAmica.location = myAmica.location .* tau'
    # Match Fortran: since scale=sbeta, and Fortran does sbeta/=Anrmk, do scale/=tau
    myAmica.scale = myAmica.scale ./ tau'
end

#Reparameterizes the parameters for the active models
function reparameterize!(myAmica::MultiModelAmica, data)
    (n, N) = size(myAmica.models[1].source_signals)
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

"Calculates sum of z. Returns N if there is just one generalized Gaussian"
function calculate_sumz(z::AbstractArray{T,3})::AbstractArray{T,2} where {T<:Real}
    (m, n, N) = size(z)

    if m == 1
        return ones(m, n) * N
    else
        sumz = sum(z, dims=3)[:, :, 1]
        sumz[sumz.<0] .= 1

        return sumz
    end
end

@views function calculate_sumz(myAmica::MultiModelAmica, h)
    return sum(myAmica.models[h].z, dims=3)
end

"Calculates densities for each sample per ICA model and per Gaussian mixture"
calculate_z!(myAmica::SingleModelAmica, i, j) = nothing
function calculate_z!(myAmica::MultiModelAmica, i, j, h)
    (m, n, N) = size(myAmica.models[0].z)

    if m > 1
        myAmica.models[h].z[j, i, :] .= myAmica.ica_weights_per_sample[h, :] .* myAmica.models[h].z[j, i, :]
    elseif m == 1
        myAmica.models[h].z[j, i, :] .= myAmica.ica_weights_per_sample[h, :]
    end
end

"Updates the Gaussian mixture location parameter. Todo: merge again with MultiModel version"
function update_location!(myAmica::SingleModelAmica{T}) where {T<:Real}
    location = myAmica.location
    shape = myAmica.shape
    zfp = myAmica.zfp
    y = myAmica.y
    scale = myAmica.scale

    (m, n, N) = size(y)

    if m <= 1
        return
    end

    dm = zeros(T, m, n)

    for k in 1:N, i in 1:n, j in 1:m
        @inbounds dm[j, i] += zfp[j, i, k] / y[j, i, k]
    end

    sum_zfp = sum(zfp, dims=3)[:, :, 1]

    dm[dm.<=0] .= 1

    a = shape .<= 2
    b = .!a .&& myAmica.kp .> 0

    # Match Fortran: since scale=sbeta, dmu_denom = scale * sum(...)
    # So: location += sum(zfp) / (scale * sum(...)) = (1/scale) * sum(zfp) / sum(...)
    location[a] .+= (1 ./ scale[a]) .* sum_zfp[a] ./ dm[a]
    location[b] .+= (1 ./ scale[b]) .* sum_zfp[b] ./ myAmica.kp[b]
end


function update_location(myAmica::MultiModelAmica, shape, zfp, y, location, scale, kp)
    # Match Fortran: since scale=sbeta, dmu_denom = scale * sum(...)
    if shape <= 2
        dm = sum(zfp ./ y)
        if dm > 0
            return location + (1 / scale) * sum(zfp) / dm
        end
    else
        if kp > 0
            return location + (1 / scale) * sum(zfp) / kp
        end
    end
    return location
end

#Updates the Gaussian mixture scale parameter
@views function update_scale!(scale::AbstractArray{T,2}, zfp::AbstractArray{T,3}, y::AbstractArray{T,3}, z::AbstractArray{T,3}, shape::AbstractArray{T,2}, y_rho::AbstractArray{T,3}) where {T<:Real}
    (m, n, N) = size(y)

    # Match Fortran: dbeta_numer and dbeta_denom
    dbeta_numer = zeros(T, m, n)
    dbeta_denom = zeros(T, m, n)

    # Accumulate numerator (always) and denominator (conditional on shape)
    @inbounds for k in 1:N, i in 1:n, j in 1:m
        # dbeta_numer = sum(z) (corresponds to Fortran's usum)
        dbeta_numer[j, i] += z[j, i, k]

        # db depends on shape value
        if shape[j, i] <= 2
            # For shape <= 2: sum(zfp * y)
            dbeta_denom[j, i] += zfp[j, i, k] * y[j, i, k]
        else
            # For shape > 2: sum(z * y_rho)
            dbeta_denom[j, i] += z[j, i, k] * y_rho[j, i, k]
        end
    end

    # Update scale: sbeta *= sqrt(dbeta_numer / db)
    @inbounds for i in 1:n, j in 1:m
        if dbeta_denom[j, i] > 0
            scale[j, i] *= sqrt(dbeta_numer[j, i] / dbeta_denom[j, i])
        end
    end
end

#Sets the initial value for the shape parameter of the GeneralizedGaussians for each Model
function initialize_shape_parameter!(myAmica::SingleModelAmica, shapelrate::LearningRate)
    myAmica.shape = shapelrate.init .* myAmica.shape
end

function initialize_shape_parameter!(myAmica::MultiModelAmica, shapelrate::LearningRate)
    initialize_shape_parameter!.(myAmica.models)
end

#Updates Gaussian mixture Parameters and mixing matrix. todo: rename since its not a loop for single model
function update_loop!(myAmica::SingleModelAmica{T}, shapelrate::LearningRate{T}, update_shape::Bool, iter::Int, do_newton::Bool, newt_start_iter::Int, lrate::LearningRate{T}) where {T<:Real}
    # Update parameters
    @timeit to "update_parameters" update_parameters!(myAmica, shapelrate, update_shape)
    # Checks for NaN in parameters before updating the mixing matrix
    if any(isnan, myAmica.source_signals) || any(isnan, myAmica.g) || any(isnan, myAmica.proportions)
        throw(AmicaNaNException())
    end
    # Update mixing matrix via Newton method
    @timeit to "newton_method" newton_method!(myAmica, iter, do_newton, newt_start_iter, lrate)
end

#Updates Gaussian mixture Parameters and mixing matrix.
function update_loop!(myAmica::MultiModelAmica, fp, y_rho, shapelrate, update_shape, iter, do_newton, newt_start_iter, lrate)
    (_, N) = size(myAmica.models[1].source_signals)
    M = size(myAmica.models, 1)

    myAmica.ica_weights_per_sample = ones(M, N)
    for h in 1:M
        #Calcutes ICA model weights
        myAmica.ica_weights_per_sample[h, :] = zeros(N)
        for i in 1:M
            myAmica.ica_weights_per_sample[h, :] = myAmica.ica_weights_per_sample[h, :] + exp.(myAmica.models[i].Lt - myAmica.models[h].Lt)
        end
        myAmica.ica_weights_per_sample[h, :] = 1 ./ myAmica.ica_weights_per_sample[h, :]
        myAmica.ica_weights[h] = sum(myAmica.ica_weights_per_sample[h, :])
        myAmica.normalized_ica_weights[h] = myAmica.ica_weights[h] / N

        #If model weight equals 0 skip update for this model
        if myAmica.normalized_ica_weights[h] == 0
            continue
        end

        g, kappa = update_parameters!(myAmica, h, fp, y_rho, shapelrate, update_shape)#todo: remove return

        #Checks for NaN in parameters before updating the mixing matrix
        if any(isnan, kappa) || any(isnan, myAmica.models[h].source_signals) || any(isnan) || any(isnan, g) || any(isnan, myAmica.models[h].proportions)
            throw(AmicaNaNException())
        end
        #Update mixing matrix via Newton method
        newton_method!(myAmica, h, iter, g, kappa, do_newton, newt_start_iter, lrate)
    end
end

#Updates Gaussian mixture parameters. It also returns g, kappa and lamda which are needed to apply the newton method.
#Todo: Save g, kappa in structure, remove return
@views function update_parameters!(myAmica::SingleModelAmica{T,ncomps,nmix}, shapelrate::LearningRate, upd_shape::Bool) where {T,ncomps,nmix}
    (m, n, N) = size(myAmica.y)


    @timeit to "update_proportions" begin
        sumz = calculate_sumz(myAmica.z)

        if m > 1
            myAmica.proportions = sumz ./ N
        end
    end

    @timeit to "ffun" ffun!(myAmica)

    @timeit to "zfp" myAmica.zfp .= myAmica.z .* myAmica.fp

    # calculate g
    @timeit to "update_g" begin
        myAmica.g .= zero(T)
        for k in 1:N, i in 1:n, j in 1:m
            @inbounds myAmica.g[i, k] += myAmica.scale[j, i] * myAmica.zfp[j, i, k]
        end
    end

    # calculate kp
    @timeit to "update_kp" begin
        myAmica.kp .= zeros(T, size(myAmica.scale))
        for k in 1:N, i in 1:n, j in 1:m
            @inbounds myAmica.kp[j, i] += myAmica.zfp[j, i, k] * myAmica.fp[j, i, k]
        end
        myAmica.kp .*= myAmica.scale
    end


    # [OK]
    @timeit to "update_location" update_location!(myAmica)
    # [OK]
    @timeit to "update_scale" update_scale!(myAmica.scale, myAmica.zfp, myAmica.y, myAmica.z, myAmica.shape, myAmica.y_rho)

    # [OK]
    if upd_shape
        @timeit to "update_shape" update_shape!(myAmica, shapelrate)
    end
end

# #Updates Gaussian mixture parameters. It also returns g, kappa and lamda which are needed to apply the newton method.
# #Todo: Save g, kappa in structure, remove return
# @views function update_parameters!(myAmica::MultiModelAmica, h, fp, y_rho, lrate_rho::LearningRate, upd_shape)
#     alpha = myAmica.models[h].proportions #todo: move into loop and add h
#     beta = myAmica.models[h].scale
#     mu = myAmica.models[h].location
#     rho = myAmica.models[h].shape

#     (n, N) = size(myAmica.models[1].source_signals)
#     m = myAmica.m
#     g = zeros(n, N)
#     kappa = zeros(n, 1)
#     zfp = zeros(m, N)


#     sumz = calculate_sumz(myAmica, h)
#     if m > 0
#         sumz[sumz.<0] .= 1
#         myAmica.models[h].z ./= sumz
#     end


#     for i in 1:n #=Threads.@threads=#
#         for j in 1:m
#             sumz = 0
#             calculate_z!(myAmica, i, j, h)
#             update_mixture_proportions!(sumz[j, i, 1], myAmica, j, i, h)
#             if sumz[j, i, 1] > 0
#                 myAmica.models[h].z[j, i, :] ./= sumz[j, i, 1]
#             else
#                 continue
#             end

#             fp[j, :] .= ffun(myAmica.models[h].y[j, i, :], rho[j, i])
#             zfp[j, :] .= myAmica.models[h].z[j, i, :] .* fp[j, :]
#             # Match Fortran: since beta=sbeta, use beta directly
#             g[i, :] .+= alpha[j, i] .* beta[j, i] .* zfp[j, :]

#             kp = beta[j, i] .* sum(zfp[j, :] .* fp[j, :])

#             kappa[i] += alpha[j, i] * kp

#             mu[j, i] = update_location(myAmica, rho[j, i], zfp[j, :], myAmica.models[h].y[j, i, :], mu[j, i], beta[j, i], kp)


#             beta[j, i] = update_scale(zfp[j, :], myAmica.models[h].y[j, i, :], beta[j, i], myAmica.models[h].z[j, i, :], rho[j, i])

#             if upd_shape == 1
#                 log_y_rho .= log.(y_rho)
#                 dr = sum(myAmica.models[h].z .* log_y_rho .* y_rho, dims=2)
#                 rho[j, i] = update_shape(rho[j, i], dr[j, i, 1], lrate_rho)
#             end
#         end
#     end
#     myAmica.models[h].proportions = alpha
#     myAmica.models[h].scale = beta
#     myAmica.models[h].location = mu
#     myAmica.models[h].shape = rho
#     return g, kappa
# end

"Updates the Gaussian mixture shape parameter"
@views function update_shape!(myAmica::SingleModelAmica{T}, shapelrate::LearningRate) where {T<:Real}
    (m, n, N) = size(myAmica.z)

    # Match Fortran: drho_numer and drho_denom
    myAmica.drho_numer .= zero(T)
    myAmica.drho_denom .= zero(T)

    # Accumulate numerator and denominator
    @timeit to "a" for k in 1:N
        # drho_numer = sum(z * y_rho * log(y_rho))
        @. myAmica.drho_numer[:, :] += myAmica.z[:, :, k] * log(myAmica.y_rho[:, :, k]) * myAmica.y_rho[:, :, k]
        # drho_denom = sum(z)
        @. myAmica.drho_denom[:, :] += myAmica.z[:, :, k]
    end


    # Update shape: rho += rholrate * (1 - (rho / digamma(1 + 1/rho)) * drho_numer / drho_denom)
    @timeit to "b" for i in 1:n, j in 1:m
        shape = myAmica.shape[j, i]
        if myAmica.drho_denom[j, i] > 0
            # Match Fortran formula
            dr2 = 1 - (shape / digamma(1 + 1 / shape)) * myAmica.drho_numer[j, i] / myAmica.drho_denom[j, i]
            if !isnan(dr2)
                myAmica.shape[j, i] += shapelrate.lrate * dr2
            end
        end
    end

    clamp!(myAmica.shape, shapelrate.minimum, shapelrate.maximum)
end

"updates the unmixed source_signals: myAmica.source_signals = myAmica.A ^ -1 * data"
function update_sources!(myAmica::SingleModelAmica{T}, data::AbstractMatrix{T}) where {T<:Real}
    qr_a = LinearAlgebra.qr(myAmica.A)
    LinearAlgebra.ldiv!(myAmica.source_signals, qr_a, data)
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

function calculate_sdLL(dLL, iter, iterwin)
    return sum(dLL[iter-iterwin+1:iter]) / iterwin
end