"""
    update_mixing!(myAmica::SingleModelAmica{T}, iter::Int, do_newton::Bool, newt_start_iter::Int,
                   lrate::LearningRate) where T<:Real

Update the mixing (unmixing) matrix using natural gradient or Newton's method.

Updates the unmixing matrix A using either the natural gradient method or Newton's method,
depending on the iteration count and the positive-definiteness of the Newton Hessian.
Newton's method is used after `newt_start_iter` iterations if `do_newton=true` and the Hessian
is positive definite. Otherwise, the natural gradient method is used with adaptive learning rates.

# Arguments
- `myAmica::SingleModelAmica{T}`: The AMICA model to update (modified in-place).
- `iter::Int`: Current iteration number.
- `do_newton::Bool`: Whether to attempt Newton's method updates.
- `newt_start_iter::Int`: Iteration at which to start using Newton's method.
- `lrate::LearningRate`: Learning rate configuration and state (modified as needed).

# Examples
```julia-repl
julia> update_mixing!(myAmica, 100, true, 50, lrate)
```

# See also
[`do_newton!`](@ref), [`calculate_lrate!`](@ref)
"""
@views function update_mixing!(
    myAmica::SingleModelAmica{T},
    iter::Int,
    do_newton::Bool,
    newt_start_iter::Int,
    lrate::LearningRate,
) where {T<:Real}
    if (do_newton && !myAmica.no_newton && iter >= newt_start_iter)
        if iter == newt_start_iter
            @info("Starting Newton ... setting numdecs to 0")
            lrate.numdecs = 0
        end

        @timeit_debug to "do_newton!" do_newton!(myAmica, lrate)
    else
        @timeit_debug to "update A" begin
            # Use natural gradient (Fortran line 2077)
            # Still ramp up learning rate but cap at lrate0
            lrate.lrate = min(
                lrate.lrate0,
                lrate.lrate + min(T(1.0) / T(lrate.newt_ramp), lrate.lrate),
            )
            myAmica.A -= lrate.lrate * myAmica.A * myAmica.dA
        end
    end
end

"""
    do_newton!(myAmica::SingleModelAmica{T}, lrate::LearningRate) where T<:Real

Perform a single Newton's method update on the unmixing matrix.

Computes the Newton Hessian matrix B and updates the unmixing matrix using Newton's method
if the Hessian is positive definite. If the Hessian is not positive definite, falls back to
the natural gradient method and sets `myAmica.no_newton = true` to avoid further Newton attempts.

# Arguments
- `myAmica::SingleModelAmica{T}`: The AMICA model to update (modified in-place).
- `lrate::LearningRate`: Learning rate configuration specifying the Newton learning rate.

# Examples
```julia-repl
julia> do_newton!(myAmica, lrate)
```

# See also
[`update_mixing!`](@ref), [`calc_b_kernel`](@ref)
"""
@views function do_newton!(
    myAmica::SingleModelAmica{T},
    lrate::LearningRate,
) where {T<:Real}
    _, n, _ = myAmica.dims

    # Build the Newton update matrix B
    B = similar(myAmica.dA)
    posdef_flag = similar(myAmica.newton_kappa, UInt8, 1)
    fill!(posdef_flag, UInt8(1))

    backend = KernelAbstractions.get_backend(myAmica.A)
    kernel! = calc_b_kernel(backend)

    @timeit_debug to "kernel" kernel!(
        B,
        posdef_flag,
        myAmica.newton_kappa,
        myAmica.newton_lambda,
        myAmica.newton_sigma2,
        myAmica.dA,
        ndrange = (n, n),
    )

    posdef = Array(posdef_flag)[1] == UInt8(1)

    # Apply update if Hessian is positive definite
    if posdef
        lrate.lrate =
            min(lrate.newtrate, lrate.lrate + min(T(1.0) / lrate.newt_ramp, lrate.lrate))
        myAmica.A -= lrate.lrate * myAmica.A * B
    else
        # Fall back to natural gradient if not positive definite (Fortran line 2074-2079)
        @info("Hessian not positive definite, using natural gradient")
        myAmica.no_newton = true

        # Still ramp up learning rate but cap at lrate0 instead of maximum
        lrate.lrate =
            min(lrate.lrate0, lrate.lrate + min(T(1.0) / T(lrate.newt_ramp), lrate.lrate))
        myAmica.A -= lrate.lrate * myAmica.A * myAmica.dA
    end
end

"""
    calc_b_kernel(B, posdef_flag, newton_kappa, newton_lambda, newton_sigma2, dA; ndrange)

GPU kernel to compute Newton Hessian matrix B and check positive-definiteness.

Computes the Newton Hessian approximation matrix B element-wise. Diagonal elements are
computed as dA[i,k] / newton_lambda[i], and off-diagonal elements use a formula combining
newton_kappa and newton_sigma2 parameters. Sets `posdef_flag[1] = 0` if any finite conditions
are violated, indicating the Hessian is not positive definite.

# Arguments
- `B::DenseArray{T,2}`: Output Hessian matrix, shape (n, n) (modified in-place).
- `posdef_flag::DenseArray{UInt8,1}`: Single-element array flag for positive-definiteness (modified).
- `newton_kappa::DenseArray{T,1}`: Newton method parameter kappa, shape (n,).
- `newton_lambda::DenseArray{T,1}`: Newton method parameter lambda, shape (n,).
- `newton_sigma2::DenseArray{T,1}`: Newton method parameter sigma squared, shape (n,).
- `dA::DenseArray{T,2}`: Gradient of unmixing matrix, shape (n, n).

# Keyword Arguments
- `ndrange`: Kernel execution range (typically `(n, n)` for the matrix dimensions).
"""
@kernel inbounds = true function calc_b_kernel(
    B::DenseArray{T,2},
    posdef_flag::DenseArray{UInt8,1},
    @Const(newton_kappa::DenseArray{T,1}),
    @Const(newton_lambda::DenseArray{T,1}),
    @Const(newton_sigma2::DenseArray{T,1}),
    @Const(dA::DenseArray{T,2})
) where {T<:Real}
    k, i = @index(Global, NTuple)

    if i == k
        # Diagonal elements
        if isfinite(newton_lambda[i]) && abs(newton_lambda[i]) > eps(T)
            B[i, k] = dA[i, k] / newton_lambda[i]
        else
            B[i, k] = zero(T)
            posdef_flag[1] = UInt8(0)
        end
    else
        # Off-diagonal elements
        sk1 = newton_sigma2[i] * newton_kappa[k]
        sk2 = newton_sigma2[k] * newton_kappa[i]

        denom = sk1 * sk2 - one(T)

        if isfinite(sk1) && isfinite(sk2) && isfinite(denom) && denom > eps(T)
            B[i, k] = (sk1 * dA[i, k] - dA[k, i]) / denom
        else
            B[i, k] = zero(T)
            posdef_flag[1] = UInt8(0)
        end
    end
end
