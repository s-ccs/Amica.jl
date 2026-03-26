"""
    remove_mean!(input::AbstractMatrix{T}) where T<:Real

Remove mean from each channel of input data matrix.

Subtracts the mean of each column from the column in-place. I.e. mean across dims=1

# Arguments
- `input::AbstractMatrix{T}`: A matrix of shape (num_samples, num_channels) containing multivariate data.

# Returns
- `mean_values`: The mean values subtracted from each channel.
"""
@views function remove_mean!(input)
    mn = mean(input, dims = 1)
    (_, n) = size(input)
    for i = 1:n
        input[:, i] .= input[:, i] .- mn[i]
    end
    return mn
end

"""
    sphering!(x::AbstractMatrix{T}) where T<:Real

Whiten data using SVD-based sphering.

Whitens the data using singular value decomposition so that the covariance matrix becomes
the identity matrix. This preprocessing step decorrelates the input data and normalizes its
variance, which improves the numerical stability and convergence of ICA algorithms.

calculates svd(x'*x), then constructs the sphering matrix S = U * Diagonal(1 ./ sqrt.(max.(S, 1e-15))) * U'
Applies the sphering to x inplace.

# Arguments
- `x::AbstractMatrix{T}`: A matrix of shape (num_samples, num_features) containing multivariate data.
  The data is modified in-place.

# Returns
- `S::AbstractMatrix{T}`: The sphering transformation matrix of shape (num_features, num_features).
  To recover the original scale, multiply sources by `S`.
"""
@views function sphering!(x)
    (N, _) = size(x)
    F = svd(x' * x / N)

    T = eltype(x)
    mineig = T(1e-15)
    safe_s = max.(F.S, mineig)
    S = F.U * Diagonal(inv.(sqrt.(safe_s))) * F.U'

    x .= x * S
    return S
end
