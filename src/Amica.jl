module Amica

    using LinearAlgebra
	using GaussianMixtures
	using Distributions
	using SpecialFunctions

    include("helper.jl")
    include("likelihood.jl")
    include("main.jl")
    include("simulate.jl")
    # Write your package code here.
    export amica
end
