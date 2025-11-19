#Updates the mixing matrix with the newton method
function newton_method!(myAmica::SingleModelAmica{T}, iter::Int, do_newton::Bool, newt_start_iter::Int, lrate::LearningRate) where {T<:Real}

    N, n, m = size(myAmica.y)

    @timeit to "a" begin
        # Calculate dA = I - g' * source_signals / N
        ArrayType = typeof(myAmica.shape)
        dA = ArrayType(I(n)) - (myAmica.g' * myAmica.source_signals) / N
    end

    if (do_newton && iter >= newt_start_iter)
        myAmica.newton_sigma2 = vec(sum(myAmica.source_signals .^ 2, dims=1) / N)

        # Build the Newton update matrix B
        B = zeros(T, n, n)
        posdef = true

        # compute B and posdef

        for i in 1:n, k in 1:n
            if i == k
                # Diagonal elements
                B[i, i] = dA[i, i] / myAmica.newton_lambda[i]
            else
                # Off-diagonal elements
                sk1 = myAmica.newton_sigma2[i] * myAmica.newton_kappa[k]
                sk2 = myAmica.newton_sigma2[k] * myAmica.newton_kappa[i]

                if sk1 * sk2 > 1.0
                    B[i, k] = (sk1 * dA[i, k] - dA[k, i]) / (sk1 * sk2 - 1.0)
                else
                    posdef = false
                end
            end
        end

        if iter == newt_start_iter
            println("Starting Newton ... setting numdecs to 0")
            lrate.numdecs = 0
        end

        # Apply update if Hessian is positive definite
        if posdef
            lrate.lrate = min(lrate.newtrate, lrate.lrate + min(T(1.0) / lrate.newt_ramp, lrate.lrate))
            myAmica.A -= lrate.lrate * myAmica.A * B
        else
            # Fall back to natural gradient if not positive definite (Fortran line 2074-2079)
            println("Hessian not positive definite, using natural gradient")

            # Still ramp up learning rate but cap at lrate0 instead of maximum
            lrate.lrate = min(lrate.lrate0, lrate.lrate + min(T(1.0) / T(lrate.newt_ramp), lrate.lrate))
            myAmica.A -= lrate.lrate * myAmica.A * dA
        end
    else
        @timeit to "b" begin

            # Use natural gradient (Fortran line 2077)
            # Still ramp up learning rate but cap at lrate0
            lrate.lrate = min(lrate.lrate0, lrate.lrate + min(T(1.0) / T(lrate.newt_ramp), lrate.lrate))
            myAmica.A -= lrate.lrate * myAmica.A * dA
        end
    end
end
