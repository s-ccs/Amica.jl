#Amica.jl is based on a MATLAB implementation of AMICA by Jason Palmer.
module Amica
    using LinearAlgebra
	using GaussianMixtures
	using Distributions
    using MKL_jll
	using SpecialFunctions
    using ProgressMeter
    using LoopVectorization
    using AppleAccelerate
    using StaticArrays
    #using ComponentArrays
    using Diagonalizations
    using LogExpFunctions
    #using MultivariateStats
    #using StatsAPI
    include("types.jl")
    include("helper.jl")
    include("likelihood.jl")
    include("parameters.jl")
    include("newton.jl")
    include("main.jl")
    
    export amica!
    export fit,fit!
    export AbstractAmica,MultiModelAmica,SingleModelAmica


    import Base.show

    function Base.show(io::Core.IO,m::SingleModelAmica)
        try
            ix = findlast(.!isnan.(m.LL))
            @show ix
            global like = string(m.LL[ix]) * " (after $(string(ix)) iterations)"
        catch
            global like = "not run"
        end
        println(io,"""
        $(typeof(m)) with:
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
