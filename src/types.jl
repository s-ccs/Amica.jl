mutable struct GGParameters{T,ncomps,nmix}
    proportions::SMatrix{nmix,ncomps,T} #source density mixture proportions
    scale::SMatrix{nmix,ncomps,T} #source density inverse scale parameter
    location::SMatrix{nmix,ncomps,T} #source density location parameter
    shape::SMatrix{nmix,ncomps,T} #source density shape paramters
end


abstract type AbstractAmica end

mutable struct SingleModelAmica{T,ncomps,nmix} <:AbstractAmica
    source_signals::Array{T,2}
    learnedParameters::GGParameters{T,ncomps,nmix}
	m::Int 		   #Number of gaussians
    A::SMatrix{ncomps,ncomps,T} # unmixing matrices for each model
	S::Array{T,2} # sphering matrix
    z::Array{T,3}
    y::Array{T,3}
    centers::Array{T,1} #model centers
    Lt::Array{T,1} #log likelihood of time point for each model ( M x N )
    LL::Array{T,1} #log likelihood over iterations todo: change to tuple 
    ldet::T
    maxiter::Int
end



mutable struct MultiModelAmica{T} <:AbstractAmica
	models::Array{SingleModelAmica{T}} #Array of SingleModelAmicas
	normalized_ica_weights 			#Model weights (normalized)
	ica_weights_per_sample 			#Model weight for each sample
	ica_weights						#Model weight for all samples
	maxiter::Int					#Number of iterations
	m::Int 							#Number of Gaussians
	LL::Array{T,1}				#Log-Likelihood
end

#Structure for Learning Rate type with initial value, minumum, maximum etc. Used for learning rate and shape lrate
using Parameters
@with_kw mutable struct LearningRate{T} 
	lrate::T = 0.1
	init::T = 0.1
	minimum::T = 0.
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
	
	centers = zeros(T,n)
	eye = Matrix(I, n, n)
	if isnothing(A)
		A = zeros(T,n,n)
		A[:,:] = eye[n] .+ 0.1*rand(n,n)
		for i in 1:n
			A[:,i] = A[:,i] / norm(A[:,i])
		end
	end

	proportions = (1/m) * ones(T,m,n)
	if isnothing(location)
		if m > 1
			location = 0.1 * randn(T,m, n)
		else
			location = zeros(T,m, n)
		end
	end
	if isnothing(scale)
		scale = ones(T,m, n) + 0.1 * randn(T,m, n)
	end
	shape = ones(T,m, n)

	y = zeros(T,m,n,N)
	
	Lt = zeros(T,N)
	z = ones(T,m,n,N)/N

	#Sets some parameters to nothing if used my MultiModel to only have them once
	if isnothing(maxiter)
		LL = nothing
		m = nothing
	else
		LL = T[]
	end
	ldet = 0.0
	source_signals = zeros(T,n,N)

	return SingleModelAmica{T,ncomps,nmix}(source_signals,GGParameters{T,ncomps,nmix}(proportions,scale,location,shape),m,A,I(size(A,1)), z,y,#=Q,=#centers,Lt,LL,ldet,maxiter)
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
		A = zeros(m,n,N)
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