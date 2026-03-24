"Temporary storage for block computation results used in update_parameters!"
mutable struct BlockAccumulators{T,Array2<:DenseArray{T,2},Array3<:DenseArray{T,3}}
    g_times_sources::Array3      # (n, n, num_threads)
    sum_z::Array3                # (n, m, num_threads)
    kp::Array3                   # (n, m, num_threads)
    dmu_numer::Array3            # (n, m, num_threads)
    dmu_denom::Array3            # (n, m, num_threads)
    dbeta_denom::Array3          # (n, m, num_threads)
    dlambda_numer::Array3        # (n, m, num_threads)
    drho_numer::Array3           # (n, m, num_threads)
    newton_sigma2::Array2        # (n, num_threads)
    Lt_accum::Array2             # (N, num_threads)
end

"Reset all accumulators to zero"
@views function reset!(acc::BlockAccumulators{T}) where {T<:Real}
    acc.g_times_sources .= zero(T)
    acc.sum_z .= zero(T)
    acc.kp .= zero(T)
    acc.dmu_numer .= zero(T)
    acc.dmu_denom .= zero(T)
    acc.dbeta_denom .= zero(T)
    acc.dlambda_numer .= zero(T)
    acc.drho_numer .= zero(T)
    acc.newton_sigma2 .= zero(T)
    acc.Lt_accum .= zero(T)
end

mutable struct SingleModelAmica{
    T,
    Array1<:DenseArray{T,1},
    Array2<:DenseArray{T,2},
    Array3<:DenseArray{T,3}
} <: AbstractAmica
    dims::NTuple{3,Int}
    block_size::Int
    num_threads::Int
    proportions::Array2                                         # source density mixture proportions
    scale::Array2                                               # source density inverse scale parameter
    location::Array2                                            # source density location parameter
    shape::Array2                                               # source density shape paramters

    A::Array2                                                   # unmixing matrix
    S::Array2                                                   # sphering matrix
    LLdetS::T                                                   # logabsdet(S)
    Lt::Array1                                                  # log likelihood of time point for each model ( M x N )
    LL::Array{T,1}                                              # log likelihood over iterations

    # --- intermediary values

    dA::Array2

    # Pre-computed values for Newton method
    newton_kappa::Array1
    newton_lambda::Array1
    newton_sigma2::Array1
    no_newton::Bool

    pools::Vector{ObjectPool{T,Array1}}                         # one pool per thread
    acc::BlockAccumulators{T,Array2,Array3}
end

"""
    SingleModelAmica([T=Float64]; nsamples, ncomps, m=3, A=nothing, location=nothing,
                     scale=nothing, block_size=10_000, num_threads=1, ArrayType=Array)

Create a single-model AMICA object.
"""
@views function SingleModelAmica(T::Type{<:Real}=Float64;
    nsamples::Int,
    ncomps::Int,
    m=3,
    A=nothing,
    location=nothing,
    scale=nothing,
    block_size=10_000,
    num_threads=1,
    ArrayType::Type{<:DenseArray}=Array
)
    N = nsamples
    n = ncomps

    # Extract array type parameters
    Array1 = ArrayType{T,1}
    Array2 = ArrayType{T,2}
    Array3 = ArrayType{T,3}

    #initialize parameters
    @timeit_debug to "init A" if isnothing(A)
        # Initialize A to match Fortran: small random ±0.005, diagonal = 1.0, then normalize
        Wtmp = rand(T, n, n)
        A = T(0.01) .* (T(0.5) .- Wtmp)  # Random values in range [-0.005, 0.005]
        for i in 1:n
            A[i, i] = T(1.0)  # Set diagonal to 1.0
            A[:, i] = A[:, i] / norm(A[:, i])  # Normalize each column
        end

        A = A |> Array2
    end

    @timeit_debug to "init proportions" begin
        proportions = Array2(undef, n, m)
        proportions .= one(T) * (1 / m)
    end

    @timeit_debug to "init location" if isnothing(location)
        # Initialize location to match Fortran: mu(j,k) = j - 1 - (m-1)/2
        # This creates centered values around 0 (e.g., -1, 0, 1 for m=3)
        location = zeros(T, n, m)
        for j in 1:m
            location[:, j] .= T(j - 1 - (m - 1) / 2)
        end
        # Add small random perturbation: ±0.05
        location .+= T(0.05) .* (T(1.0) .- T(2.0) .* rand(T, n, m))
    end

    @timeit_debug to "init scale" if isnothing(scale)
        # Initialize scale to match Fortran: 1.0 + 0.1*(0.5 - random[0,1])
        # This gives values in range [0.95, 1.05]
        scale = ones(T, n, m) .+ T(0.1) .* (T(0.5) .- rand(T, n, m))
    end


    @timeit_debug to "init pools" pools = [ObjectPool{T,Array1}(block_size * n * m, 7) for _ in 1:num_threads]

    return SingleModelAmica{T,Array1,Array2,Array3}(
        (N, n, m),
        block_size,
        num_threads,
        Array2(proportions),                       # proportions
        Array2(scale),                             # scale
        Array2(location),                          # location
        Array2(ones(T, n, m)),                     # shape
        A,                                           # A
        Array2(undef, n, n),                         # S
        zero(T),                                     # LLdetS
        Array1(undef, N),                            # Lt
        Array1(undef, 0),                            # LL
        Array2(undef, n, n),                         # dA
        Array1(undef, n),                            # newton_kappa
        Array1(undef, n),                            # newton_lambda
        Array1(undef, n),                            # newton_sigma2
        false,                                       # no_newton
        pools,
        BlockAccumulators{T,Array2,Array3}(
            Array3(undef, n, n, num_threads),           # g_times_sources
            Array3(undef, n, m, num_threads),           # sum_z
            Array3(undef, n, m, num_threads),           # kp
            Array3(undef, n, m, num_threads),           # dmu_numer
            Array3(undef, n, m, num_threads),           # dmu_denom
            Array3(undef, n, m, num_threads),           # dbeta_denom
            Array3(undef, n, m, num_threads),           # dlambda_numer
            Array3(undef, n, m, num_threads),           # drho_numer
            Array2(undef, n, num_threads),              # newton_sigma2
            Array2(undef, N, num_threads)               # Lt_accum
        ))
end
