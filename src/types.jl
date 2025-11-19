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
mutable struct LearningRate{T}
    lrate::T
    init::T
    minimum::T
    maximum::T
    natural_rate::T
    decreaseFactor::T

    function LearningRate{T}(;
        lrate::T=0.1,
        init::T=0.1,
        minimum::T=0.0,
        maximum::T=1.0,
        natural_rate::T=0.1,
        decreaseFactor::T=0.5
    ) where {T}
        new(lrate, init, minimum, maximum, natural_rate, decreaseFactor)
    end
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



struct AmicaProportionsZeroException <: Exception
end

struct AmicaNaNException <: Exception
end