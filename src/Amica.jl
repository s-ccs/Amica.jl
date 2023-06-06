module Amica
    using LinearAlgebra
	using GaussianMixtures
	using Distributions
	using SpecialFunctions

    include("types.jl")
    include("helper.jl")
    include("likelihood.jl")
    include("main.jl")
    include("simulate.jl")
    # Write your package code here.
    export amica
    export fit,fit!
    export AbstractAmica,MultiModelAmica
end
