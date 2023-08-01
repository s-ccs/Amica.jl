mutable struct GGParameters
	prop::AbstractArray{Float64} #source density mixture proportions
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
	source_signals::AbstractArray
	learnedParameters::GGParameters
	M::Integer #number of ica models
	n::Integer #number of data channels
	m::Integer #number of gaussians
	N::Integer #number of timesteps
    A::AbstractArray #unmixing matrices for each model
	z::AbstractArray
	y::AbstractArray
	Q::AbstractArray
	centers::AbstractArray #model centers
	Lt::AbstractMatrix #log likelihood of time point for each model ( M x N )
	LL::AbstractMatrix #log likelihood over iterations todo: change to tuple
	ldet::AbstractArray
	proportions::AbstractMatrix #model proportions
	maxiter::Integer #maximum number of iterations
end

mutable struct MultiModelAmica <:AbstractAmica
	singleModel::SingleModelAmica
	#Models::ComponentArray
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
function SingleModelAmica(data::AbstractArray{T}; m=3, M=1, maxiter=500, A=nothing, mu=nothing, beta=nothing, kwargs...) where {T<:Real}
	# M, m, maxiter, update_rho, mindll, iterwin, do_newton, remove_mean
	(n, N) = size(data)
	

	#initialize parameters
	
	centers = zeros(n,M)
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

	proportions = (1/M) * ones(M,1)
	alpha = (1/m) * ones(m,n,M)
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
	rho = ones(m, n, M)

	
	y = zeros(n,N,m,M)
	
	Q = zeros(m,N)
	
	Lt = zeros(M,N)
	z = ones(n,N,m,M)/N


	#originally initialized inside the loop
	LL = zeros(1,maxiter)
	ldet = zeros(M)
	source_signals = zeros(n,N,M)


	return SingleModelAmica(source_signals,GGParameters(alpha,beta,mu,rho),M,n,m,N,A,z,y,Q,centers,Lt,LL,ldet,proportions,maxiter)
end

function MultiModelAmica(data::Array; m=3, M=1, maxiter=500, A=nothing, mu=nothing, beta=nothing, kwargs...)
	multiModel = SingleModelAmica(data; m, M, maxiter, A, mu, beta, kwargs...)
	return MultiModelAmica(multiModel)
end


import Base.getproperty
# Base.getproperty(x::AbstractAmica, s::Symbol) = Base.getproperty(x, Val(s))
# Base.getproperty(x::AbstractAmica, ::Val{s}) where s = getfield(x, s)

# Base.getproperty(m::AbstractAmica, ::Val{:X}) = size(m.source_signals,4)


function Base.getproperty(multiModel::MultiModelAmica, prop::Symbol)
    if prop in fieldnames(SingleModelAmica) && !(prop in fieldnames(MultiModelAmica))
        return getfield(multiModel.singleModel, prop)
    else
        return getfield(multiModel, prop)
    end
end

struct AmicaProportionsZeroException <: Exception
end