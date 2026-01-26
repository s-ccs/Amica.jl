"Updates the mixing matrix"
function update_mixing!(myAmica::SingleModelAmica{T}, iter::Int, do_newton::Bool, newt_start_iter::Int, lrate::LearningRate) where {T<:Real}
    if (do_newton && iter >= newt_start_iter)
        if iter == newt_start_iter
            println("Starting Newton ... setting numdecs to 0")
            lrate.numdecs = 0
        end

        @timeit to "do_newton!" do_newton!(myAmica, lrate)
    else
        @timeit to "update A" begin
            # Use natural gradient (Fortran line 2077)
            # Still ramp up learning rate but cap at lrate0
            lrate.lrate = min(lrate.lrate0, lrate.lrate + min(T(1.0) / T(lrate.newt_ramp), lrate.lrate))
            myAmica.A -= lrate.lrate * myAmica.A * myAmica.dA
        end
    end
end

"Perform the newton method"
function do_newton!(myAmica::SingleModelAmica{T}, lrate::LearningRate) where {T<:Real}
    N, n = size(myAmica.z)

    # Build the Newton update matrix B
    B = similar(myAmica.dA)
    posdef = true

    backend = KernelAbstractions.get_backend(myAmica.source_signals)
    kernel! = calc_b_kernel(backend)

    @timeit to "kernel" kernel!(B, posdef, myAmica.newton_kappa, myAmica.newton_lambda, myAmica.newton_sigma2, myAmica.dA, ndrange=(n, n))

    # Apply update if Hessian is positive definite
    if posdef
        lrate.lrate = min(lrate.newtrate, lrate.lrate + min(T(1.0) / lrate.newt_ramp, lrate.lrate))
        myAmica.A -= lrate.lrate * myAmica.A * B
    else
        # Fall back to natural gradient if not positive definite (Fortran line 2074-2079)
        println("Hessian not positive definite, using natural gradient")

        # Still ramp up learning rate but cap at lrate0 instead of maximum
        lrate.lrate = min(lrate.lrate0, lrate.lrate + min(T(1.0) / T(lrate.newt_ramp), lrate.lrate))
        myAmica.A -= lrate.lrate * myAmica.A * myAmica.dA
    end
end

@kernel inbounds = true unsafe_indices = true function calc_b_kernel(
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
