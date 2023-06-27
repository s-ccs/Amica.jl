mutable struct GGParameters
	α::AbstractArray{Float64} #source density mixture proportions
	β::AbstractArray{Float64} #source density inverse scale parameter
	μ::AbstractArray{Float64} #source density location parameter
	ρ::AbstractArray{Float64} #source density shape paramters
end

mutable struct MoreParameters
	kappa::AbstractArray
end

abstract type AbstractAmica end

mutable struct MultiModelAmica <:AbstractAmica
	#moreParameters::MoreParameters
	source_signals::AbstractArray
	learnedParameters::GGParameters
	M::Integer #number of ica models
	n::Integer
	m::Integer
	N::Integer
    A::AbstractArray #unmixing matrices for each model
	z::AbstractArray
	y::AbstractArray
	Q::AbstractArray
	centers::AbstractArray #model centers
	Lt::AbstractMatrix #log likelihood of time point for each model ( M x N )
	LL::AbstractMatrix #log likelihood over iterations
	ldet::AbstractArray
	proportions::AbstractMatrix
	maxiter::Integer #maximum number of iterations
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


function MultiModelAmica(data::Array; m=3, M=1, maxiter=500, A=nothing, mu=nothing, beta=nothing, kwargs...)
	# M, m, maxiter, update_rho, mindll, iterwin, do_newton, remove_mean
	(n, N) = size(data)
	

	#initialize parameters
	
	centers = zeros(n,M)
	eye = Matrix{Float64}(I, n, n)
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

	return MultiModelAmica(source_signals,GGParameters(alpha,beta,mu,rho),M,n,m,N,A,z,y,Q,centers,Lt,LL,ldet,proportions,maxiter)
end

import Base.getproperty
Base.getproperty(x::AbstractAmica, s::Symbol) = Base.getproperty(x, Val(s))
Base.getproperty(x::AbstractAmica, ::Val{s}) where s = getfield(x, s)

Base.getproperty(m::AbstractAmica, ::Val{:X}) = size(m.source_signals,4)

# struct SinglemodelAmica <:AbstractAmica
#     A::AbstractArray
# end

struct AmicaProportionsZeroException <: Exception
end