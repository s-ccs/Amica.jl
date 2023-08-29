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
	n::Integer #number of data channels
	m::Integer #number of gaussians
	N::Integer #number of timesteps
    A::AbstractArray #unmixing matrices for each model
	z::AbstractArray
	y::AbstractArray
	Q::AbstractArray
	centers::AbstractArray #model centers
	Lt::AbstractVector #log likelihood of time point for each model ( M x N )
	LL::AbstractVector #log likelihood over iterations todo: change to tuple 
	ldet::Float64
	maxiter::Integer #maximum number of iterations
end

mutable struct MultiModelAmica <:AbstractAmica
	#singleModel::SingleModelAmica
	models::Array{SingleModelAmica} #Array of SingleModelAmicas
	model_proportions::AbstractMatrix #model proportions
	ldet::AbstractArray #currently in both amicas todo: change that
	v
	vsum
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
	
	Q = zeros(m,N)
	
	Lt = zeros(N)
	z = ones(n,N,m)/N


	#originally initialized inside the loop
	LL = zeros(maxiter) #todo: TUPEL
	ldet = 0.0
	source_signals = zeros(n,N)


	return SingleModelAmica(source_signals,GGParameters(alpha,beta,mu,rho),n,m,N,A,z,y,Q,centers,Lt,LL,ldet,maxiter)
end

function MultiModelAmica(data::Array; m=3, M=2, maxiter=500, A=nothing, mu=nothing, beta=nothing, kwargs...)
	# multiModel = SingleModelAmica(data; m, M, maxiter, A, mu, beta, kwargs...)
	# return MultiModelAmica(multiModel)
	models = Array{SingleModelAmica}(undef, M)
	model_proportions = (1/M) * ones(M,1)
	(n, N) = size(data)
	ldet = zeros(M)
	v = ones(N)
	vsum = zeros(M)
	for h in 1:M
		models[h] = SingleModelAmica(data; m, maxiter, A, mu, beta, kwargs...)
	end
	return MultiModelAmica(models,model_proportions,ldet,v,vsum)
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
function Base.getproperty(multiModel::MultiModelAmica, prop::Symbol)
    if prop in fieldnames(SingleModelAmica) && !(prop in fieldnames(MultiModelAmica))
        return getfield(multiModel.models[1], prop)
    else
        return getfield(multiModel, prop)
    end
end

struct AmicaProportionsZeroException <: Exception
end

struct AmicaNaNException <: Exception
end