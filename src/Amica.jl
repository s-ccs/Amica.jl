#Amica.jl is based on a MATLAB implementation of AMICA by Jason Palmer.
module Amica

using LinearAlgebra
using SpecialFunctions
using TimerOutputs
using Statistics
using StatsAPI
import StatsAPI: fit
using KernelAbstractions
using ProgressMeter
using PrecompileTools: @setup_workload, @compile_workload

const to = TimerOutput()
const NAN_CHECK_ACTIVE = false

"""Abstract type to be implemented by AMICA model variants."""
abstract type AbstractAmica end

include("object_pool.jl")
include("types.jl")
include("learning_rate.jl")
include("numerics.jl")
include("preprocessing.jl")
include("diagnostics.jl")
include("block_processing.jl")
include("parameter_updates.jl")
include("mixing.jl")
include("main.jl")

export amica!
export fit
export recover_sources
export AbstractAmica, SingleModelAmica

# precompilation
@compile_workload begin
    Amica.fit(
        SingleModelAmica,
        zeros(Float32, 3_000, 24),
        m = 3,
        maxiter = 1,
        newt_start_iter = 0,
        show_progress = false,
    )
    Amica.fit(
        SingleModelAmica,
        zeros(Float64, 3_000, 24),
        m = 3,
        maxiter = 1,
        newt_start_iter = 0,
        show_progress = false,
    )
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
    - dims (N, n, m): $(m.dims)
    - likelihood: $(like)
""",
    )
end

end
