mutable struct GGParameters
	proportions::AbstractArray{Float64} #source density mixture proportions
	scale::AbstractArray{Float64} #source density inverse scale parameter
	location::AbstractArray{Float64} #source density location parameter
	shape::AbstractArray{Float64} #source density shape paramters
end

#not in use
mutable struct MoreParameters
	kappa::AbstractArray
end

abstract type AbstractAmica end


mutable struct SingleModelAmica <:AbstractAmica
	#moreParameters::MoreParameters
	source_signals
	learnedParameters::GGParameters
	m::Integer #number of gaussians
    A::AbstractArray #unmixing matrices for each model
	z::AbstractArray
	y::AbstractArray
	Q::Union{AbstractArray, Nothing} #1xN
	centers::AbstractArray #model centers
	Lt::AbstractVector #log likelihood of time point for each model ( M x N )
	LL::Union{AbstractVector, Nothing} #log likelihood over iterations todo: change to tuple 
	ldet::Float64
	maxiter::Union{Int, Nothing} #maximum number of iterations, can be nothing because it's not needed for multimodel
end

mutable struct MultiModelAmica <:AbstractAmica
	#singleModel::SingleModelAmica
	models::Array{SingleModelAmica} #Array of SingleModelAmicas
	model_proportions::AbstractMatrix #model proportions
	ldet::AbstractArray #currently in both amicas todo: change that
	v
	vsum
	maxiter::Int
	m::Int #Number of Gaussians
	LL::AbstractVector
	Q
end

using Parameters
@with_kw mutable struct LearningRate
	lrate::Real = 0.1
	init::Float64 = 0.1
	minimum::Float64 = 0.
	maximum::Float64 = 1.0
	natural_rate::Float64 = 0.1
	decreaseFactor::Float64 = 0.5
end

#todo: rename gg parameters
function SingleModelAmica(data::AbstractArray{T}; m=3, maxiter=500, A=nothing, mu=nothing, beta=nothing, kwargs...) where {T<:Real}
	# M, m, maxiter, update_rho, mindll, iterwin, do_newton, remove_mean
	(n, N) = size(data)
	

	#initialize parameters
	
	centers = zeros(n)
	eye = Matrix(I, n, n) #todo: check if necessary
	if isnothing(A)
		A = zeros(n,n)
		A[:,:] = eye[n] .+ 0.1*rand(n,n)
		for i in 1:n
			A[:,i] = A[:,i] / norm(A[:,i])
		end
	end

	alpha = (1/m) * ones(m,n)
	if isnothing(mu)
		if m > 1
			mu = 0.1 * randn(m, n)
		else
			mu = zeros(m, n)
		end
	end
	if isnothing(beta)
		beta = ones(m, n) + 0.1 * randn(m, n)
	end
	rho = ones(m, n)

	
	y = zeros(n,N,m)
	
	
	Lt = zeros(N)
	z = ones(n,N,m)/N


	#Sets some parameters to nothing to only have them once in MultiModel
	if isnothing(maxiter)
		LL = nothing #todo: check if this works in Multimodel. maxiter will be given as nothing by MultiModel constructor
		Q = nothing
	else
		LL = Float64[]
		Q = zeros(m,N)
	end
	ldet = 0.0
	source_signals = zeros(n,N)


	return SingleModelAmica(source_signals,GGParameters(alpha,beta,mu,rho),m,A,z,y,Q,centers,Lt,LL,ldet,maxiter)
end

function MultiModelAmica(data::Array; m=3, M=2, maxiter=500, A=nothing, mu=nothing, beta=nothing, kwargs...)
	# multiModel = SingleModelAmica(data; m, M, maxiter, A, mu, beta, kwargs...)
	# return MultiModelAmica(multiModel)
	models = Array{SingleModelAmica}(undef, M)
	model_proportions = (1/M) * ones(M,1)
	(n, N) = size(data)
	ldet = zeros(M)
	v = ones(M,N)
	vsum = zeros(M)
	LL = Float64[]
	Q = zeros(m,N)

	#This part only exists to allow for initial values to be set by the user. They are still required to have the old format (something, something, M)
	eye = Matrix(I, n, n) #todo: check if necessary
	if isnothing(A)
		A = zeros(n,n,M)
		for h in 1:M
			A[:,:,h] = eye[n] .+ 0.1*rand(n,n)
			for i in 1:n
				A[:,i,h] = A[:,i,h] / norm(A[:,i,h])
			end
		end
	end

	if isnothing(mu)
		if m > 1
			mu = 0.1 * randn(m, n, M)
		else
			mu = zeros(m, n, M)
		end
	end

	if isnothing(beta)
		beta = ones(m, n, M) + 0.1 * randn(m, n, M)
	end

	for h in 1:M
		models[h] = SingleModelAmica(data; m, maxiter=nothing, A=A[:,:,h], mu=mu[:,:,h], beta=beta[:,:,h], kwargs...)
	end
	return MultiModelAmica(models,model_proportions,ldet,v,vsum,maxiter,m,LL,Q)
end


import Base.getproperty
# Base.getproperty(x::AbstractAmica, s::Symbol) = Base.getproperty(x, Val(s))
# Base.getproperty(x::AbstractAmica, ::Val{s}) where s = getfield(x, s)

# Base.getproperty(m::AbstractAmica, ::Val{:X}) = size(m.source_signals,4)


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