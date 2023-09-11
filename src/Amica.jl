module Amica
    using LinearAlgebra
	using GaussianMixtures
	using Distributions
	using SpecialFunctions
    using ProgressMeter
    using ComponentArrays
    using Diagonalizations
    #using MultivariateStats
    #using StatsAPI
    include("types.jl")
    include("helper.jl")
    include("likelihood.jl")
    include("parameters.jl")
    include("newton.jl")
    include("main.jl")
    include("simulate.jl")
    # Write your package code here.
    export amica!
    export fit,fit!
    export AbstractAmica,MultiModelAmica,SingleModelAmica


    import Base.show

    function Base.show(io::Core.IO,m::SingleModelAmica)
        try
            global like = m.LL[findlast(m.LL .!= 0)]
        catch
            global like = "not run"
        end
        println(io,"""
        Amica with:
            - signal-size: $(size(m.source_signals))
            - likelihood: $(like) 
        """)
    end

    function Base.show(io::Core.IO,m::MultiModelAmica)
        try
            global like = m.LL[findlast(m.LL .!= 0)]
        catch
            global like = "not run"
        end
        println(io,"""
        Amica with:
            - models: $(length(m.models))
            - signal-size: $(size(m.models[1].source_signals))
            - likelihood: $(like) 
        """)
    end

    
end
