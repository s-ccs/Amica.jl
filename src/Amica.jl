#Amica.jl is based on a MATLAB implementation of AMICA by Jason Palmer.
module Amica

using LinearAlgebra
using SpecialFunctions
using TimerOutputs
using Statistics
using KernelAbstractions
using PrecompileTools: @setup_workload, @compile_workload


const to = TimerOutput()
const NAN_CHECK_ACTIVE = false

abstract type AbstractAmica end

include("object_pool.jl")
include("types.jl")
include("helper.jl")
include("parameters.jl")
include("mixing.jl")
include("main.jl")

export amica!
export fit, fit!
export AbstractAmica, SingleModelAmica

# precompilation
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

end
