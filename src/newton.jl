#Updates the mixing matrix with the newton method
function newton_method!(myAmica::SingleModelAmica{T}, iter::Int, do_newton::Bool, newt_start_iter::Int, lrate::LearningRate) where {T<:Real}
    N, n, m = size(myAmica.y)

    # Calculate dA = I - g' * source_signals / N
    dA = I(n) - (myAmica.g' * myAmica.source_signals) / N

    if do_newton && (iter >= newt_start_iter)
        # Calculate sigma2 = sum(source_signals^2) / N
        sigma2 = vec(sum(myAmica.source_signals .^ 2, dims=2) / N)

        # Calculate baralpha (proportions), same as proportions in single model case
        baralpha = myAmica.proportions  # (n, m)

        # Initialize kappa and lambda
        kappa = zeros(T, n)
        lambda = zeros(T, n)

        # Calculate kappa and lambda by summing over mixture components
        for i in 1:n
            for j in 1:m
                # dkap represents E[fp * fp] per mixture component
                dkap = myAmica.kp[i, j]
                kappa[i] += baralpha[i, j] * dkap

                # lambda includes both the variance term and location shift
                # dlambda = sum(z * (fp*y - 1)^2) / sum(z)
                # Note: sum(z[:,i,j]) = baralpha[i,j] * N, so we divide by (baralpha*N) not just N
                sum_z = baralpha[i, j] * N
                dlambda = sum(myAmica.z[:, i, j] .* (myAmica.fp[:, i, j] .* myAmica.y[:, i, j] .- 1.0) .^ 2) / sum_z

                lambda[i] += baralpha[i, j] * (dlambda + dkap * myAmica.location[i, j]^2)
            end
        end

        # Build the Newton update matrix B
        B = zeros(T, n, n)
        posdef = true

        for i in 1:n
            for k in 1:n
                if i == k
                    # Diagonal elements
                    B[i, i] = dA[i, i] / lambda[i]
                else
                    # Off-diagonal elements
                    sk1 = sigma2[i] * kappa[k]
                    sk2 = sigma2[k] * kappa[i]

                    if sk1 * sk2 > 1.0
                        B[i, k] = (sk1 * dA[i, k] - dA[k, i]) / (sk1 * sk2 - 1.0)
                    else
                        posdef = false
                    end
                end
            end
        end

        # Apply update if Hessian is positive definite
        if posdef
            myAmica.A -= lrate.lrate * myAmica.A * B
        else
            # Fall back to natural gradient if not positive definite
            myAmica.A -= lrate.natural_rate * myAmica.A * dA
        end
    else
        # Use natural gradient
        myAmica.A -= lrate.natural_rate * myAmica.A * dA
    end
end
