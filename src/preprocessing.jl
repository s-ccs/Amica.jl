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
    input .= input .- mn
    return mn
end

"""
    sphering!(x::AbstractMatrix{T}; ncomps=nothing) where T<:Real

Whiten data using SVD-based sphering, optionally reducing the dimensionality
with PCA first (as in MNE's `ICA(n_components=k)` or eeglab's `runica 'pca',k`).

Whitens the data using singular value decomposition so that the covariance matrix becomes
the identity matrix. This preprocessing step decorrelates the input data and normalizes its
variance, which improves the numerical stability and convergence of ICA algorithms.

calculates svd(x'*x), then constructs the sphering matrix S = U * Diagonal(1 ./ sqrt.(max.(S, 1e-15))) * U'
Applies the sphering to x inplace.

When `ncomps` is less than the number of channels, only the `ncomps` leading
principal components are kept: `S = U[:, 1:k] * Diagonal(1 ./ sqrt.(max.(s, 1e-15))[1:k])`
then has shape (num_features, ncomps) and the data is projected onto that
reduced subspace. The whitened data still has identity covariance in the
retained subspace.

# Arguments
- `x::AbstractMatrix{T}`: A matrix of shape (num_samples, num_features) containing multivariate data.
  The data is modified in-place when no dimensionality reduction is performed.
- `ncomps::Union{Nothing,Int} = nothing`: Number of principal components to keep.
  `nothing` (default) or a value ≥ the number of channels keeps the full rank.

# Returns
- `S::AbstractMatrix{T}`: The sphering transformation matrix of shape
  (num_features, num_features) for the full rank, or (num_features, ncomps)
  after PCA reduction. To recover the original scale, multiply sources by `S`.
- `lldet::T`: Log absolute determinant of the whitening transform, i.e.
  `-0.5 * sum(log.(eigenvalues))` over the retained components.
- `y::AbstractMatrix{T}`: The transformed data. The same array as `x` for the
  full rank; a new matrix with `ncomps` columns after PCA reduction (a matrix
  cannot be shrunk in place).
"""
@views function sphering!(x; ncomps = nothing)
    (N, n) = size(x)
    ncomps = isnothing(ncomps) ? nothing : clamp(ncomps, 1, n)
    F = svd(x' * x / N)

    T = eltype(x)
    mineig = T(1e-15)
    safe_s = max.(F.S, mineig)
    if isnothing(ncomps) || ncomps >= n
        S = F.U * Diagonal(inv.(sqrt.(safe_s))) * F.U'
        lldet = logabsdet(S |> Array)[1]
        x .= x * S
        return S, lldet, x
    else
        # PCA reduction: keep the leading principal components
        k = ncomps
        S = F.U[:, 1:k] * Diagonal(inv.(sqrt.(safe_s[1:k])))
        lldet = -sum(log.(safe_s[1:k])) / 2
        return S, lldet, x * S # fewer columns than the input, cannot update in place
    end
end
