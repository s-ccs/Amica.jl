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
end

# struct SinglemodelAmica <:AbstractAmica
#     A::AbstractArray
# end

#todo:where Q??? g?? v?? kappa?? proportions?? lambda??
#hardcoding, softcoding und mehrere models
#ersten zwei zeilen in loop in funktion machen
#tausend variablen auch in struct? konstruktor dafür?
#m,n,N,M nicht immer übergeben sondern in fkt ausrechnen