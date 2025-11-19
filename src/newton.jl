#Updates the mixing matrix with the newton method
function newton_method!(myAmica::SingleModelAmica{T}, iter::Int, g, kappa, do_newton::Bool, newt_start_iter::Int, lrate::LearningRate) where {T<:Real}

    (n, N) = size(myAmica.source_signals)

    # Match Fortran: divide by N (dgm_numer which equals all_blks)
    dA = Matrix{Float64}(I, n, n) - (g * myAmica.source_signals') / N

    if (do_newton == 1) && (iter > newt_start_iter)
        lrate = lrate.lrate
        sigma2 = sum(myAmica.source_signals .^ 2, dims=2) / N #is probably called sigma2 cause always squared
        B = zeros(n, n)
        bflag = false



        for k in 1:N, i in 1:n
            lambda = zero(T)
            for j in 1:m
                lambda += gg.proportions[j, i] * ((myAmica.z[j, i, k] * (myAmica.fp[j, i, k] * myAmica.y[j, i, k])^2) + (gg.location[j, i] .^ 2 .* kp[j, i]) / N)
            end

            if any(isnan, lambda)
                throw(AmicaNaNException())
            end

            if i == k
                B[i, i] = dA[i, i] / lambda
            else
                denom = kappa[i] * kappa[k] * sigma2[i] * sigma2[k] - 1
                if denom > 0
                    B[i, k] = (-kappa[k] * sigma2[i] * dA[i, k] + dA[k, i]) / denom
                else
                    bflag = true
                end
            end
        end

        if (bflag == false)
            myAmica.A -= lrate * myAmica.A * B
        end
    else
        lnatrate = lrate.natural_rate
        myAmica.A -= lnatrate * myAmica.A * dA
    end
end

@views function newton_method!(myAmica::MultiModelAmica, h, iter, g, kappa, do_newton, newt_start_iter, lrate::LearningRate)

    lnatrate = lrate.natural_rate
    lrate = lrate.lrate
    (n, N) = size(myAmica.models[1].source_signals)

    sigma2 = myAmica.models[h].source_signals .^ 2 * myAmica.ica_weights_per_sample[h, :] / myAmica.ica_weights[h]

    # Match Fortran: divide by N (dgm_numer which equals all_blks)
    dA = Matrix{Float64}(I, n, n) - (g * myAmica.models[h].source_signals') / N
    bflag = 0
    B = zeros(n, n)

    for i in 1:n
        for k in 1:N
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