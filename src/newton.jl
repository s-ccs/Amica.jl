#Updates the mixing matrix with the newton method
function newton_method!(myAmica::Union{<:CuSingleModelAmica{T},<:SingleModelAmica{T}}, iter::Int, g, kappa, do_newton::Bool, newt_start_iter::Int, lrate::LearningRate, lambda::AbstractArray{T,1}) where {T<:Real}

    lnatrate = lrate.natural_rate
    lrate = lrate.lrate
    (n, N) = size(myAmica.source_signals)

    sigma2 = sum(myAmica.source_signals .^ 2, dims=2) / N #is probably called sigma2 cause always squared

    dA = I - g * myAmica.source_signals'
    bflag = false
    B = zeros(T, n, n)

    for i in 1:n
        for k = 1:n
            if i == k
                B[i, i] = dA[i, i] / (lambda[i])
            else
                denom = kappa[i] * kappa[k] * sigma2[i] * sigma2[k] - 1
                if denom > 0
                    B[i, k] = (-kappa[k] * sigma2[i] * dA[i, k] + dA[k, i]) / denom
                else
                    bflag = true
                end
            end
        end
    end
    if (bflag == false) && (do_newton == 1) && (iter > newt_start_iter)
        myAmica.A += lrate * myAmica.A * B
    else
        myAmica.A -= lnatrate * myAmica.A * dA
    end
end

@views function newton_method!(myAmica::MultiModelAmica, h, iter, g, kappa, do_newton, newt_start_iter, lrate::LearningRate, lambda)

    lnatrate = lrate.natural_rate
    lrate = lrate.lrate
    M = size(myAmica.models)
    (n, N) = size(myAmica.models[1].source_signals)

    sigma2 = myAmica.models[h].source_signals .^ 2 * myAmica.ica_weights_per_sample[h, :] / myAmica.ica_weights[h]

    dA = Matrix{Float64}(I, n, n) - g * myAmica.models[h].source_signals'
    bflag = 0
    B = zeros(n, n)

    for i in 1:n
        for k = 1:n
            if i == k
                B[i, i] = dA[i, i] / (lambda[i])
            else
                denom = kappa[i] * kappa[k] * sigma2[i] * sigma2[k] - 1
                if denom > 0
                    B[i, k] = (-kappa[k] * sigma2[i] * dA[i, k] + dA[k, i]) / denom
                else
                    bflag = 1
                end
            end
        end
    end
    if (bflag == 0) && (do_newton == 1) && (iter > newt_start_iter)
        myAmica.models[h].A = myAmica.models[h].A + lrate * myAmica.models[h].A * B
    else
        myAmica.models[h].A = myAmica.models[h].A - lnatrate * myAmica.models[h].A * dA
    end
end