mutable struct SingleModelAmica{
    T,
    Array1<:DenseArray{T,1},
    Array2<:DenseArray{T,2},
    Array3<:DenseArray{T,3}
} <: AbstractAmica
    source_signals::Array2                                      # unmixed source signals (A^-1 * x)

    proportions::Array2                                         # source density mixture proportions
    scale::Array2                                               # source density inverse scale parameter
    location::Array2                                            # source density location parameter
    shape::Array2                                               # source density shape paramters

    A::Array2                                                   # unmixing matrix
    S::Array2                                                   # sphering matrix
    LLdetS::T                                                   # logabsdet(S)
    z::Array3
    y::Array3
    Lt::Array1                                                  # log likelihood of time point for each model ( M x N )
    LL::Array{T,1}                                              # log likelihood over iterations

    # --- intermediary values

    y_rho::Array3                                               # abs(y)^(rho-1)
    scratch1::Array3
    scratch2::Array3
    dA::Array2

    # Pre-computed values for Newton method (using scale before update)
    newton_kappa::Array1
    newton_lambda::Array1
    newton_sigma2::Array1
end

"Data type for AMICA with just one ICA model."
function SingleModelAmica(T::Type{<:Real}=Float64;
    nsamples::Int,
    ncomps::Int,
    m=3,
    A=nothing,
    location=nothing,
    scale=nothing,
    ArrayType::Type{<:DenseArray}=Array
)
    N = nsamples
    n = ncomps

    #initialize parameters
    @timeit to "init A" if isnothing(A)
        # Initialize A to match Fortran: small random ±0.005, diagonal = 1.0, then normalize
        Wtmp = rand(T, n, n)
        A = T(0.01) .* (T(0.5) .- Wtmp)  # Random values in range [-0.005, 0.005]
        for i in 1:n
            A[i, i] = T(1.0)  # Set diagonal to 1.0
            A[:, i] = A[:, i] / norm(A[:, i])  # Normalize each column
        end
    end

    @timeit to "init proportions" proportions = (1 / m) * ones(T, n, m)
    @timeit to "init location" if isnothing(location)
        # Initialize location to match Fortran: mu(j,k) = j - 1 - (m-1)/2
        # This creates centered values around 0 (e.g., -1, 0, 1 for m=3)
        location = zeros(T, n, m)
        for j in 1:m
            location[:, j] .= T(j - 1 - (m - 1) / 2)
        end
        # Add small random perturbation: ±0.05
        location .+= T(0.05) .* (T(1.0) .- T(2.0) .* rand(T, n, m))
    end
    @timeit to "init scale" if isnothing(scale)
        # Initialize scale to match Fortran: 1.0 + 0.1*(0.5 - random[0,1])
        # This gives values in range [0.95, 1.05]
        scale = ones(T, n, m) .+ T(0.1) .* (T(0.5) .- rand(T, n, m))
    end

    # Extract array type parameters
    Array1 = ArrayType{T,1}
    Array2 = ArrayType{T,2}
    Array3 = ArrayType{T,3}

    return SingleModelAmica{T,Array1,Array2,Array3}(
        zeros(T, BLOCK_SIZE, n) |> Array2,           # source_signals
        proportions |> Array2,                       # proportions
        scale |> Array2,                             # scale
        location |> Array2,                          # location
        ones(T, n, m) |> Array2,                     # shape
        A |> Array2,                                 # A
        Matrix{T}(I(size(A, 1))) |> Array2,          # S
        zero(T),                                     # LLdetS
        (ones(T, BLOCK_SIZE, n, m) / N) |> Array3,   # z
        zeros(T, BLOCK_SIZE, n, m) |> Array3,        # y
        zeros(T, N) |> Array1,                       # Lt
        T[] |> Array1,                               # LL
        zeros(T, BLOCK_SIZE, n, m) |> Array3,        # y_rho
        zeros(T, BLOCK_SIZE, n, m) |> Array3,        # scratch1
        zeros(T, BLOCK_SIZE, n, m) |> Array3,        # scratch2
        zeros(T, n, n) |> Array2,                    # dA
        zeros(T, n) |> Array1,                       # newton_kappa
        zeros(T, n) |> Array1,                       # newton_lambda
        zeros(T, n) |> Array1,                       # newton_sigma2
    )
end
