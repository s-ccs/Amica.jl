#Normalizes source density location parameter (mu), scale parameter (beta) and model centers
function reparameterize!(myAmica::SingleModelAmica, data)

    tau = norm.(eachcol(myAmica.A))
    @debug size(tau) tau
    myAmica.A = myAmica.A ./ tau'
    myAmica.learnedParameters.location = myAmica.learnedParameters.location .* tau'
    myAmica.learnedParameters.scale = myAmica.learnedParameters.scale ./ tau' .^ 2

end

#Reparameterizes the parameters for the active models
function reparameterize!(myAmica::MultiModelAmica, data)
    (n, N) = size(myAmica.models[1].source_signals)
    M = size(myAmica.models, 1)

    for h = 1:M
        mu = myAmica.models[h].learnedParameters.location
        beta = myAmica.models[h].learnedParameters.scale

        if myAmica.normalized_ica_weights[h] == 0
            continue
        end
        for i in 1:n
            tau = norm(myAmica.models[h].A[:, i])
            myAmica.models[h].A[:, i] = myAmica.models[h].A[:, i] / tau
            mu[:, i] = mu[:, i] * tau
            beta[:, i] = beta[:, i] / tau^2
        end

        if M > 1
            cnew = data * myAmica.ica_weights_per_sample[h, :] / (sum(myAmica.ica_weights_per_sample[h, :])) #todo: check why v not inverted
            for i in 1:n
                Wh = pinv(myAmica.models[h].A[:, :])
                mu[:, i] = mu[:, i] .- Wh[i, :]' * (cnew - myAmica.models[h].centers[:])
            end
            myAmica.models[h].centers = cnew
        end
        myAmica.models[h].learnedParameters.location .= mu
        myAmica.models[h].learnedParameters.scale .= beta
    end
end

#Calculates sum of z. Returns N if there is just one generalized Gaussian
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

#Calculates densities for each sample per ICA model and per Gaussian mixture
calculate_z!(myAmica::SingleModelAmica, i, j) = nothing
function calculate_z!(myAmica::MultiModelAmica, i, j, h)
    if myAmica.m > 1
        myAmica.models[h].z[j, i, :] .= myAmica.ica_weights_per_sample[h, :] .* myAmica.models[h].z[j, i, :]
    elseif myAmica.m == 1
        myAmica.models[h].z[j, i, :] .= myAmica.ica_weights_per_sample[h, :]
    end
end

#Updates the Gaussian mixture location parameter. Todo: merge again with MultiModel version
function update_location!(location::AbstractArray{T,2}, shape::AbstractArray{T,2}, zfp::AbstractArray{T,3}, y::AbstractArray{T,3}, scale::AbstractArray{T,2}, kp::AbstractArray{T,2}) where {T<:Real}
    (m, n, N) = size(y)

    if m <= 1
        return
    end

    dm = zeros(T, m, n)

    for k = 1:N, i = 1:n, j = 1:m
        @inbounds dm[j, i] += zfp[j, i, k] / y[j, i, k]
    end

    sum_zfp = sum(zfp, dims=3)[:, :, 1]

    dm[dm.<=0] .= 1

    a = shape .<= 2
    b = .!a .&& kp .> 0

    location[a] .+= (1 ./ sqrt.(scale[a])) .* sum_zfp[a] ./ dm[a]
    location[b] .+= sqrt.(scale[b]) .* sum_zfp[b] ./ kp[b]
end


function update_location(myAmica::MultiModelAmica, shape, zfp, y, location, scale, kp)
    if shape <= 2
        dm = sum(zfp ./ y)
        if dm > 0
            return location + (1 / sqrt(scale)) * sum(zfp) / dm
        end
    else
        if kp > 0
            return location + sqrt(scale) * sum(zfp) / kp
        end
    end
    return location
end

#Updates the Gaussian mixture scale parameter
@views function update_scale!(scale::AbstractArray{T,2}, zfp::AbstractArray{T,3}, y::AbstractArray{T,3}, z::AbstractArray{T,3}, shape::AbstractArray{T,2}, y_rho::AbstractArray{T,3}) where {T<:Real}
    (m, n, N) = size(y)

    db = zeros(T, m, n)

    @inbounds for k = 1:N, i = 1:n, j = 1:m
        if shape[j, i] <= 2
            db[j, i] += zfp[j, i, k] * y[j, i, k]
        else
            db[j, i] += z[j, i, k] * y_rho[j, i, k]
        end
    end

    @inbounds for i = 1:n, j = 1:m
        if shape[j, i] <= 2
            if db[j, i] >= 0
                scale[j, i] /= db[j, i]
            end
        else
            db[j, i] *= shape[j, i] * db[j, i]^(-2 / shape[j, i])
        end
    end
end

#Sets the initial value for the shape parameter of the GeneralizedGaussians for each Model
function initialize_shape_parameter!(myAmica::SingleModelAmica, shapelrate::LearningRate)
    myAmica.learnedParameters.shape = shapelrate.init .* myAmica.learnedParameters.shape
end

function initialize_shape_parameter!(myAmica::MultiModelAmica, shapelrate::LearningRate)
    initialize_shape_parameter!.(myAmica.models)
end

#Updates Gaussian mixture Parameters and mixing matrix. todo: rename since its not a loop for single model
function update_loop!(myAmica::SingleModelAmica{T}, shapelrate::LearningRate{T}, update_shape::Bool, iter::Int, do_newton::Bool, newt_start_iter::Int, lrate::LearningRate{T}) where {T<:Real}
    #Update parameters
    g, kappa = update_parameters!(myAmica, shapelrate, update_shape)
    #Checks for NaN in parameters before updating the mixing matrix
    if any(isnan, kappa) || any(isnan, myAmica.source_signals) || any(isnan, myAmica.lambda) || any(isnan, g) || any(isnan, myAmica.learnedParameters.proportions)
        throw(AmicaNaNException())
    end
    #Update mixing matrix via Newton method
    newton_method!(myAmica, iter, g, kappa, do_newton, newt_start_iter, lrate, myAmica.lambda)
end

#Updates Gaussian mixture Parameters and mixing matrix.
function update_loop!(myAmica::MultiModelAmica, fp, lambda, y_rho, shapelrate, update_shape, iter, do_newton, newt_start_iter, lrate)
    (n, N) = size(myAmica.models[1].source_signals)
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

        g, kappa, lambda = update_parameters!(myAmica, h, fp, lambda, y_rho, shapelrate, update_shape)#todo: remove return

        #Checks for NaN in parameters before updating the mixing matrix
        if any(isnan, kappa) || any(isnan, myAmica.models[h].source_signals) || any(isnan, lambda) || any(isnan, g) || any(isnan, myAmica.models[h].learnedParameters.proportions)
            throw(AmicaNaNException())
        end
        #Update mixing matrix via Newton method
        newton_method!(myAmica, h, iter, g, kappa, do_newton, newt_start_iter, lrate, lambda)
    end
end

#Updates Gaussian mixture parameters. It also returns g, kappa and lamda which are needed to apply the newton method.
#Todo: Save g, kappa, lambda in structure, remove return
@views function update_parameters!(myAmica::SingleModelAmica{T,ncomps,nmix}, shapelrate::LearningRate, upd_shape::Bool) where {T,ncomps,nmix}
    (m, n, N) = size(myAmica.y)

    gg = myAmica.learnedParameters


    sumz = calculate_sumz(myAmica.z)

    if m > 0
        myAmica.z ./= sumz
    end

    if m > 1
        myAmica.learnedParameters.proportions = sumz ./ N
    end

    ffun!(myAmica.fp, myAmica.y, gg.shape)
    myAmica.zfp .= myAmica.z .* myAmica.fp

    # calculate g
    myAmica.g .= zero(T)
    for k = 1:N, i = 1:n, j = 1:m
        @inbounds myAmica.g[i, k] += gg.proportions[j, i] * sqrt(gg.scale[j, i]) * myAmica.zfp[j, i, k]
    end

    # calculate kp
    kp = zeros(size(gg.scale))
    for k = 1:N, i = 1:n, j = 1:m
        @inbounds kp[j, i] += myAmica.zfp[j, i, k] * myAmica.fp[j, i, k]
    end
    kp .*= gg.scale

    # calculate lambda
    for k = 1:N, i = 1:n, j = 1:m
        @inbounds myAmica.lambda[i] += gg.proportions[j, i] * ((myAmica.z[j, i, k] * (myAmica.fp[j, i, k] * myAmica.y[j, i, k])^2) + (gg.location[j, i] .^ 2 .* kp[j, i]) / N)
    end

    kappa = sum(gg.proportions .* kp, dims=1)[1, :]

    update_location!(gg.location, gg.shape, myAmica.zfp, myAmica.y, gg.scale, kp)
    update_scale!(gg.scale, myAmica.zfp, myAmica.y, myAmica.z, gg.shape, myAmica.y_rho)


    if upd_shape
        update_shape!(gg.shape, myAmica.z, myAmica.y_rho, shapelrate)
    end

    return myAmica.g, kappa
end

#Updates Gaussian mixture parameters. It also returns g, kappa and lamda which are needed to apply the newton method.
#Todo: Save g, kappa, lambda in structure, remove return
@views function update_parameters!(myAmica::MultiModelAmica, h, fp, y_rho, lambda, lrate_rho::LearningRate, upd_shape)
    alpha = myAmica.models[h].learnedParameters.proportions #todo: move into loop and add h
    beta = myAmica.models[h].learnedParameters.scale
    mu = myAmica.models[h].learnedParameters.location
    rho = myAmica.models[h].learnedParameters.shape

    (n, N) = size(myAmica.models[1].source_signals)
    m = myAmica.m
    g = zeros(n, N)
    kappa = zeros(n, 1)
    zfp = zeros(m, N)


    sumz = calculate_sumz(myAmica, h)
    if m > 0
        sumz[sumz.<0] .= 1
        myAmica.models[h].z ./= sumz
    end


    for i in 1:n #=Threads.@threads=#
        for j in 1:m
            sumz = 0
            calculate_z!(myAmica, i, j, h)
            update_mixture_proportions!(sumz[j, i, 1], myAmica, j, i, h)
            if sumz[j, i, 1] > 0
                myAmica.models[h].z[j, i, :] ./= sumz[j, i, 1]
            else
                continue
            end

            fp[j, :] .= ffun(myAmica.models[h].y[j, i, :], rho[j, i])
            zfp[j, :] .= myAmica.models[h].z[j, i, :] .* fp[j, :]
            g[i, :] .+= alpha[j, i] .* sqrt(beta[j, i]) .* zfp[j, :]

            kp = beta[j, i] .* sum(zfp[j, :] .* fp[j, :])

            kappa[i] += alpha[j, i] * kp

            lambda[i] += alpha[j, i] .* (sum(myAmica.models[h].z[j, i, :] .* (fp[j, :] .* myAmica.models[h].y[j, i, :] .- 1) .^ 2) .+ mu[j, i]^2 .* kp)
            mu[j, i] = update_location(myAmica, rho[j, i], zfp[j, :], myAmica.models[h].y[j, i, :], mu[j, i], beta[j, i], kp)


            beta[j, i] = update_scale(zfp[j, :], myAmica.models[h].y[j, i, :], beta[j, i], myAmica.models[h].z[j, i, :], rho[j, i])

            if upd_shape == 1
                log_y_rho = optimized_log(y_rho)
                dr = sum(myAmica.models[h].z .* log_y_rho .* y_rho, dims=2)
                rho[j, i] = update_shape(rho[j, i], dr[j, i, 1], lrate_rho)
            end
        end
    end
    myAmica.models[h].learnedParameters.proportions = alpha
    myAmica.models[h].learnedParameters.scale = beta
    myAmica.models[h].learnedParameters.location = mu
    myAmica.models[h].learnedParameters.shape = rho
    return g, kappa
end

#Updates the Gaussian mixture shape parameter
@views function update_shape!(shape::AbstractArray{T,2}, z::AbstractArray{T,3}, y_rho::AbstractArray{T,3}, shapelrate::LearningRate) where {T<:Real}
    (m, n, N) = size(z)

    dr = zeros(T, m, n)
    log_y_rho = optimized_log(y_rho)

    for k = 1:N, i = 1:n, j = 1:m
        @inbounds dr[j, i] = z[j, i, k] * log_y_rho[j, i, k] * y_rho[j, i, k]
    end


    for i in 1:n
        for j in 1:m
            _shape = shape[j, i]
            if _shape > 2
                dr2 = digamma(1 + 1 / _shape) / _shape - dr[j, i]
                if !isnan(dr2)
                    shape[j, i] += 0.5 * dr2
                end
            else
                dr2 = 1 - _shape * dr[j, i] / digamma(1 + 1 / _shape)
                if !isnan(dr2)
                    shape[j, i] += shapelrate.lrate * dr2
                end
            end
        end
    end

    clamp!(shape, shapelrate.minimum, shapelrate.maximum)
end

#Calculates determinant of mixing Matrix A (with log). first log-likelihood part of L = |A| * p(sources)
function calculate_ldet!(myAmica::SingleModelAmica)
    #myAmica.ldet = -log(abs(det(myAmica.A)))
    myAmica.ldet = -logabsdet(myAmica.A)[1]
    @debug :ldet myAmica.ldet
end

function calculate_ldet!(myAmica::MultiModelAmica)
    for h in 1:length(myAmica.models)
        myAmica.models[h].ldet = -log(abs(det(myAmica.models[h].A)))
    end
end

#Updates source singal estimations by unmixing the data
function update_sources!(myAmica::SingleModelAmica{T}, data::AbstractMatrix{T}) where {T<:Real}
    #myAmica.source_signals .= myAmica.A \ data#
    #myAmica.source_signals .= myAmica.A \ data#
    #ldiv!(myAmica.source_signals,myAmica.A,data)
    #lu!(myAmica.A)

    LinearAlgebra.ldiv!(myAmica.source_signals, LinearAlgebra.qr(myAmica.A), data)
    #myAmica.source_signals = pinv(myAmica.A) * data
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