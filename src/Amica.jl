#Amica.jl is based on a MATLAB implementation of AMICA by Jason Palmer.
module Amica

using LinearAlgebra
using SpecialFunctions
using TimerOutputs
using Statistics
using Parameters
using KernelAbstractions
using Atomix

const to = TimerOutput()
const NAN_CHECK_ACTIVE = true

abstract type AbstractAmica end

#using MultivariateStats
#using StatsAPI
include("single_model_amica.jl")
include("util.jl")
include("types.jl")
include("helper.jl")
include("likelihood.jl")
include("parameters.jl")
include("mixing.jl")
include("main.jl")

export amica!
export fit, fit!
export AbstractAmica, MultiModelAmica, SingleModelAmica


import Base.show

function Base.show(io::Core.IO, m::SingleModelAmica)
    global like = "not run"
    try
        ix = findlast(.!isnan.(m.LL))
        @show ix
        global like = string(m.LL[ix]) * " (after $(string(ix)) iterations)"
    catch
    end
    println(
        io,
        """
$(typeof(m)) with:
    - signal-size: $(size(m.source_signals))
    - likelihood: $(like)
"""
    )
end

function Base.show(io::Core.IO, m::MultiModelAmica)
    global like = "not run"
    try
        global like = m.LL[findlast(m.LL .!= 0)]
    catch
    end
    println(
        io,
        """
Amica with:
    - models: $(length(m.models))
    - signal-size: $(size(m.models[1].source_signals))
    - likelihood: $(like)
"""
    )
end


end
