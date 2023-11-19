mutable struct GGParameters{T}
    proportions::Array{T, 2} #source density mixture proportions
    scale::Array{T, 2} #source density inverse scale parameter
    location::Array{T, 2} #source density location parameter
    shape::Array{T, 2} #source density shape paramters
end


abstract type AbstractAmica end

mutable struct SingleModelAmica{T} <:AbstractAmica
    source_signals::Array{T,2}
    learnedParameters::GGParameters{T}
	m::Union{Integer, Nothing} 		   #Number of gaussians
    A::Array{T,2} #unmixing matrices for each model
    z::Array{T,3}
    y::Array{T,3}
    centers::Array{T} #model centers
    Lt::Array{Float64} #log likelihood of time point for each model ( M x N )
    LL::Array{T} #log likelihood over iterations todo: change to tuple 
    ldet::Float64
    maxiter::Int
end


mutable struct MultiModelAmica <:AbstractAmica
	models::Array{SingleModelAmica} #Array of SingleModelAmicas
	normalized_ica_weights 			#Model weights (normalized)
	ica_weights_per_sample 			#Model weight for each sample
	ica_weights						#Model weight for all samples
	maxiter::Int					#Number of iterations
	m::Int 							#Number of Gaussians
	LL::AbstractVector				#Log-Likelihood
end

#Structure for Learning Rate type with initial value, minumum, maximum etc. Used for learning rate and shape lrate
using Parameters
@with_kw mutable struct LearningRate
	lrate::Real = 0.1
	init::Float64 = 0.1
	minimum::Float64 = 0.
	maximum::Float64 = 1.0
	natural_rate::Float64 = 0.1
	decreaseFactor::Float64 = 0.5
end

#Data type for AMICA with just one ICA model. todo: rename gg parameters
function SingleModelAmica(data::AbstractArray{T}; m=3, maxiter=500, A=nothing, location=nothing, scale=nothing, kwargs...) where {T<:Real}
	(n, N) = size(data)
	#initialize parameters
	
	centers = zeros(n)
	eye = Matrix(I, n, n)
	if isnothing(A)
		A = zeros(n,n)
		A[:,:] = eye[n] .+ 0.1*rand(n,n)
		for i in 1:n
			A[:,i] = A[:,i] / norm(A[:,i])
		end
	end

	proportions = (1/m) * ones(m,n)
	if isnothing(location)
		if m > 1
			location = 0.1 * randn(m, n)
		else
			location = zeros(m, n)
		end
	end
	if isnothing(scale)
		scale = ones(m, n) + 0.1 * randn(m, n)
	end
	shape = ones(m, n)

	y = zeros(n,N,m)
	
	Lt = zeros(N)
	z = ones(n,N,m)/N

	#Sets some parameters to nothing if used my MultiModel to only have them once
	if isnothing(maxiter)
		LL = nothing
		m = nothing
	else
		LL = Float64[]
	end
	ldet = 0.0
	source_signals = zeros(n,N)

	return SingleModelAmica{T}(source_signals,GGParameters{T}(proportions,scale,location,shape),m,A,z,y,#=Q,=#centers,Lt,LL,ldet,maxiter)
end

#Data type for AMICA with multiple ICA models
function MultiModelAmica(data::Array; m=3, M=2, maxiter=500, A=nothing, location=nothing, scale=nothing, kwargs...)
	models = Array{SingleModelAmica}(undef, M) #Array of SingleModelAmica opjects
	normalized_ica_weights = (1/M) * ones(M,1)
	(n, N) = size(data)
	ica_weights_per_sample = ones(M,N)
	ica_weights = zeros(M)
	LL = Float64[]

	#This part only exists to allow for initial values to be set by the user. They are still required to have the old format (something x something x M)
	eye = Matrix(I, n, n)
	if isnothing(A)
		A = zeros(n,n,M)
		for h in 1:M
			A[:,:,h] = eye[n] .+ 0.1*rand(n,n)
			for i in 1:n
				A[:,i,h] = A[:,i,h] / norm(A[:,i,h])
			end
		end
	end

	if isnothing(location)
		if m > 1
			location = 0.1 * randn(m, n, M)
		else
			location = zeros(m, n, M)
		end
	end

	if isnothing(scale)
		scale = ones(m, n, M) + 0.1 * randn(m, n, M)
	end

	for h in 1:M
		models[h] = SingleModelAmica(data; m, maxiter=nothing, A=A[:,:,h], location=location[:,:,h], scale=scale[:,:,h], kwargs...)
	end
	return MultiModelAmica(models,normalized_ica_weights,ica_weights_per_sample,ica_weights,maxiter,m,LL#=,Q=#)
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