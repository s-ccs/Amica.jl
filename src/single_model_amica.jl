@with_kw_noshow mutable struct SingleModelAmica{
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
    LL::Array1                                                  # log likelihood over iterations todo: change to tuple

    # --- intermediary values

    y_rho::Array3                                               # abs(y)^rho
    fp::Array3

    zfp::Array3                                                 # z * fp
    g::Array2

    kp::Array2

    drho_numer::Array2
    drho_denom::Array2
end

"Data type for AMICA with just one ICA model."
function SingleModelAmica(
    data::DenseArray{T,2};
    m=3,
    A=nothing,
    location=nothing,
    scale=nothing,
    ArrayType::Type{<:DenseArray}=Array
) where {T<:Real}
    (n, N) = size(data)

    #initialize parameters

    if isnothing(A)
        # Initialize A to match Fortran: small random ±0.005, diagonal = 1.0, then normalize
        Wtmp = rand(T, n, n)
        A = T(0.01) .* (T(0.5) .- Wtmp)  # Random values in range [-0.005, 0.005]
        for i in 1:n
            A[i, i] = T(1.0)  # Set diagonal to 1.0
            A[:, i] = A[:, i] / norm(A[:, i])  # Normalize each column
        end
    end

    proportions = (1 / m) * ones(T, m, n)
    if isnothing(location)
        # Initialize location to match Fortran: mu(j,k) = j - 1 - (m-1)/2
        # This creates centered values around 0 (e.g., -1, 0, 1 for m=3)
        location = zeros(T, m, n)
        for j in 1:m
            location[j, :] .= T(j - 1 - (m - 1) / 2)
        end
        # Add small random perturbation: ±0.05
        location .+= T(0.05) .* (T(1.0) .- T(2.0) .* rand(T, m, n))
    end
    if isnothing(scale)
        # Initialize scale to match Fortran: 1.0 + 0.1*(0.5 - random[0,1])
        # This gives values in range [0.95, 1.05]
        scale = ones(T, m, n) .+ T(0.1) .* (T(0.5) .- rand(T, m, n))
    end

    # Extract array type parameters
    Array1 = ArrayType{T,1}
    Array2 = ArrayType{T,2}
    Array3 = ArrayType{T,3}

    return SingleModelAmica{T,Array1,Array2,Array3}(
        source_signals=zeros(T, n, N) |> Array2,
        proportions=proportions |> Array2,
        scale=scale |> Array2,
        location=location |> Array2,
        shape=ones(T, m, n) |> Array2,
        A=A |> Array2,
        S=Matrix{T}(I(size(A, 1))) |> Array2,
        LLdetS=zero(T),
        z=(ones(T, m, n, N) / N) |> Array3,
        y=zeros(T, m, n, N) |> Array3,
        Lt=zeros(T, N) |> Array1,
        LL=T[] |> Array1,
        y_rho=zeros(T, m, n, N) |> Array3,
        fp=zeros(T, m, n, N) |> Array3,
        zfp=zeros(T, m, n, N) |> Array3,
        g=zeros(T, n, N) |> Array2,
        kp=zeros(T, m, n) |> Array2,
        drho_numer=zeros(T, m, n) |> Array2,
        drho_denom=zeros(T, m, n) |> Array2,
    )
end
