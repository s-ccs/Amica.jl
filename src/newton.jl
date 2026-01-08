
@kernel function newton_kernel!(
    B::DenseArray{T,2},
    posdef::Bool,
    @Const(newton_kappa::DenseArray{T,1}),
    @Const(newton_lambda::DenseArray{T,1}),
    @Const(newton_sigma2::DenseArray{T,1}),
    @Const(dA::DenseArray{T,2})
) where T<:Real
    k, i = @index(Global, NTuple)

    if i == k
        # Diagonal elements
        B[i, k] = dA[i, k] / newton_lambda[i]
    else
        # Off-diagonal elements
        sk1 = newton_sigma2[i] * newton_kappa[k]
        sk2 = newton_sigma2[k] * newton_kappa[i]

        if sk1 * sk2 > T(1.0)
            B[i, k] = (sk1 * dA[i, k] - dA[k, i]) / (sk1 * sk2 - T(1.0))
        else
            posdef = false
        end
    end
end


# Updates the mixing matrix with the newton method
function newton_method!(myAmica::SingleModelAmica{T}, iter::Int, do_newton::Bool, newt_start_iter::Int, lrate::LearningRate) where {T<:Real}

    N, n, m = size(myAmica.y)

    @timeit to "a" begin
        # Calculate dA = I - g' * source_signals / N
        ArrayType = typeof(myAmica.shape)
        dA = ArrayType(I(n)) - (myAmica.g' * myAmica.source_signals) / N
    end

    if (do_newton && iter >= newt_start_iter)
        myAmica.newton_sigma2 .= sum(myAmica.source_signals .^ 2, dims=1)[1, :] ./ N

        # Build the Newton update matrix B
        B = similar(dA)
        posdef = true
        if iter == newt_start_iter
            println("Starting Newton ... setting numdecs to 0")
            lrate.numdecs = 0
        end



        backend = KernelAbstractions.get_backend(myAmica.source_signals)
        kernel! = newton_kernel!(backend)


        @timeit to "kernel" kernel!(B, posdef, myAmica.newton_kappa, myAmica.newton_lambda, myAmica.newton_sigma2, dA, ndrange=(n, n))

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
