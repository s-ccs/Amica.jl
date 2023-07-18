module Amica
    using LinearAlgebra
	using GaussianMixtures
	using Distributions
	using SpecialFunctions
    using ProgressMeter
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

    function Base.show(io::Core.IO,m::AbstractAmica)
        try
            global like = m.LL[findlast(m.LL .!= 0)]
        catch
            global like = "not run"
        end
        println(io,"""
        Amica with:
            - models: $(m.M)
            - signal-size: $(size(m.source_signals))
            - likelihood: $(like) 
        """)
    end

    
end
