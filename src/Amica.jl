#Amica.jl is based on a MATLAB implementation of AMICA by Jason Palmer.
module Amica

using LinearAlgebra
using SpecialFunctions
using TimerOutputs
using Statistics
using Parameters
using KernelAbstractions
using Atomix
using PrecompileTools: @setup_workload, @compile_workload


const to = TimerOutput()
const NAN_CHECK_ACTIVE = false

abstract type AbstractAmica end

include("object_pool.jl")
include("single_model_amica.jl")
include("types.jl")
include("helper.jl")
include("likelihood.jl")
include("parameters.jl")
include("mixing.jl")
include("main.jl")

export amica!
export fit, fit!
export AbstractAmica, MultiModelAmica, SingleModelAmica

@compile_workload begin
    Amica.fit(SingleModelAmica, zeros(Float32, 3_000, 24), m=3, maxiter=1, newt_start_iter=0, show_progress=false)
    Amica.fit(SingleModelAmica, zeros(Float64, 3_000, 24), m=3, maxiter=1, newt_start_iter=0, show_progress=false)
end

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
