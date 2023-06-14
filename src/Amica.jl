module Amica
    using LinearAlgebra
	using GaussianMixtures
	using Distributions
	using SpecialFunctions
    using ProgressMeter
    include("types.jl")
    include("helper.jl")
    include("likelihood.jl")
    include("main.jl")
    include("simulate.jl")
    # Write your package code here.
    export amica!
    export fit,fit!
    export AbstractAmica,MultiModelAmica


    import Base.show

    function Base.show(io::Core.IO,m::AbstractAmica)
        println(io,"""
        Amica with:
            - models: $(m.M)
            - signal-size: $(size(m.source_signals))
            - likelihood: $(m.Lt) 
        """)
    end

    
end
