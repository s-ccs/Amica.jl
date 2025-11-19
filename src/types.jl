mutable struct GGParameters{T,ncomps,nmix}
    proportions::Array{T,2} #source density mixture proportions
    scale::Array{T,2} #source density inverse scale parameter
    location::Array{T,2} #source density location parameter
    shape::Array{T,2} #source density shape paramters
end


abstract type AbstractAmica end

mutable struct SingleModelAmica{T,ncomps,nmix} <: AbstractAmica
    "unmixed source signals (A^-1 * x)"
    source_signals::Array{T,2}

    learnedParameters::GGParameters{T,ncomps,nmix}
    m::Int    #Number of gaussians
    A::Array{T,2} # unmixing matrix
    S::Array{T,2} # sphering matrix
    LLdetS::T
    z::Array{T,3}
    y::Array{T,3}
    centers::Array{T,1} #model centers
    Lt::Array{T,1} #log likelihood of time point for each model ( M x N )
    LL::Array{T,1} #log likelihood over iterations todo: change to tuple 
    ldet::T
    maxiter::Int

    # --- intermediary values
    # precalculated abs(y)^rho
    y_rho::Array{T,3}
    log_y_rho::Array{T,3}
    lambda::Array{T,1}
    fp::Array{T,3}
    # z * fp
    zfp::Array{T,3}
    g::Array{T,2}
    Q::Array{T,3}
    drho_numer::Array{T,2}
    drho_denom::Array{T,2}

    u_intermed::Array{T,4}
end



mutable struct MultiModelAmica{T} <: AbstractAmica
    models::Array{SingleModelAmica{T}} #Array of SingleModelAmicas
    normalized_ica_weights #Model weights (normalized)
    ica_weights_per_sample #Model weight for each sample
    ica_weights#Model weight for all samples
    maxiter::Int#Number of iterations
    m::Int #Number of Gaussians
    LL::Array{T,1}#Log-Likelihood
end

#Structure for Learning Rate type with initial value, minumum, maximum etc. Used for learning rate and shape lrate
using Parameters
@with_kw mutable struct LearningRate{T}
    lrate::T = 0.1
    init::T = 0.1
    minimum::T = 0.0
    maximum::T = 1.0
    natural_rate::T = 0.1
    decreaseFactor::T = 0.5
end

#Data type for AMICA with just one ICA model. todo: rename gg parameters
function SingleModelAmica(data::AbstractArray{T}; m=3, maxiter=500, A=nothing, location=nothing, scale=nothing, kwargs...) where {T<:Real}
    (n, N) = size(data)
    ncomps = n
    nmix = m
    #initialize parameters

    centers = zeros(T, n)
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
    shape = ones(T, m, n)

    y = zeros(T, m, n, N)
    y_rho = zeros(T, m, n, N)
    log_y_rho = zeros(T, m, n, N)

    drho_numer = zeros(T, m, n)
    drho_denom = zeros(T, m, n)

    Lt = zeros(T, N)
    z = ones(T, m, n, N) / N

    #Sets some parameters to nothing if used my MultiModel to only have them once
    if isnothing(maxiter)
        LL = nothing
        m = nothing
    else
        LL = T[]
    end
    ldet = 0.0
    source_signals = zeros(T, n, N)
    lambda = zeros(T, n)
    fp = zeros(T, m, n, N)
    zfp = zeros(T, m, n, N)
    Q = zeros(T, m, n, N)
    g = zeros(T, n, N)
    u_intermed = zeros(T, m, n, N, m)

    return SingleModelAmica{T,ncomps,nmix}(
        source_signals,
        GGParameters{T,ncomps,nmix}(proportions, scale, location, shape),
        m,
        A,
        I(size(A, 1)),
        zero(T),
        z,
        y,
        centers,
        Lt,
        LL,
        ldet,
        maxiter,
        y_rho,
        log_y_rho,
        lambda,
        fp,
        zfp,
        g,
        Q,
        drho_numer,
        drho_denom,
        u_intermed,
    )
end

#Data type for AMICA with multiple ICA models
function MultiModelAmica(data::Array; m=3, M=2, maxiter=500, A=nothing, location=nothing, scale=nothing, kwargs...)
    models = Array{SingleModelAmica}(undef, M) #Array of SingleModelAmica opjects
    normalized_ica_weights = (1 / M) * ones(M, 1)
    (n, N) = size(data)
    ica_weights_per_sample = ones(M, N)
    ica_weights = zeros(M)
    LL = Float64[]

    #This part only exists to allow for initial values to be set by the user. They are still required to have the old format (something x something x M)
    if isnothing(A)
        A = zeros(n, n, M)
        for h in 1:M
            # Initialize A to match Fortran: small random ±0.005, diagonal = 1.0, then normalize
            Wtmp = rand(n, n)
            A[:, :, h] = 0.01 .* (0.5 .- Wtmp)  # Random values in range [-0.005, 0.005]
            for i in 1:n
                A[i, i, h] = 1.0  # Set diagonal to 1.0
                A[:, i, h] = A[:, i, h] / norm(A[:, i, h])  # Normalize each column
            end
        end
    end

    if isnothing(location)
        # Initialize location to match Fortran: mu(j,k) = j - 1 - (m-1)/2
        # This creates centered values around 0 (e.g., -1, 0, 1 for m=3)
        location = zeros(m, n, M)
        for h in 1:M
            for j in 1:m
                location[j, :, h] .= j - 1 - (m - 1) / 2
            end
            # Add small random perturbation: ±0.05
            location[:, :, h] .+= 0.05 .* (1.0 .- 2.0 .* rand(m, n))
        end
    end

    if isnothing(scale)
        # Initialize scale to match Fortran: 1.0 + 0.1*(0.5 - random[0,1])
        # This gives values in range [0.95, 1.05]
        scale = ones(m, n, M) .+ 0.1 .* (0.5 .- rand(m, n, M))
    end

    for h in 1:M
        models[h] = SingleModelAmica(data; m, maxiter=nothing, A=A[:, :, h], location=location[:, :, h], scale=scale[:, :, h], kwargs...)
    end
    return MultiModelAmica(models, normalized_ica_weights, ica_weights_per_sample, ica_weights, maxiter, m, LL) #=,Q=#
end


# import Base.getproperty
#  Base.getproperty(x::AbstractAmica, s::Symbol) = Base.getproperty(x, Val(s))
#  Base.getproperty(x::AbstractAmica, ::Val{s}) where s = getfield(x, s)

#  Base.getproperty(m::AbstractAmica, ::Val{:N}) = size(m.Lt,1)
#  Base.getproperty(m::AbstractAmica, ::Val{:n}) = size(m.A,1)
#  Base.getproperty(m::MultiModelAmica, ::Val{:M}) = length(m.models)
#  Base.getproperty(m::SingleModelAmica, ::Val{:M}) = 1


# function Base.getproperty(multiModel::MultiModelAmica, prop::Symbol)
#     if prop in fieldnames(SingleModelAmica) && !(prop in fieldnames(MultiModelAmica))
#         return getfield(multiModel.singleModel, prop)
#     else
#         return getfield(multiModel, prop)
#     end
# end

#currently not necessary
# function Base.getproperty(multiModel::MultiModelAmica, prop::Symbol)
#     if prop in fieldnames(SingleModelAmica) && !(prop in fieldnames(MultiModelAmica))
#         return getfield(multiModel.models[1], prop)
#     else
#         return getfield(multiModel, prop)
#     end
# end

struct AmicaProportionsZeroException <: Exception
end

struct AmicaNaNException <: Exception
end