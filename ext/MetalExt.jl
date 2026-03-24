module MetalExt

using Metal, LinearAlgebra, SpecialFunctions, Amica
using PrecompileTools: @setup_workload, @compile_workload

import Base.Broadcast

function LinearAlgebra.:\(a::MtlArray{T}, b::MtlArray{T}) where T
    return ((a |> Array{T}) \ (b |> Array{T})) |> MtlArray{T}
end

function LinearAlgebra.inv(a::MtlArray{T}) where T
    return LinearAlgebra.inv(a |> Array{T}) |> MtlArray{T}
end

function Broadcast.broadcasted(::Metal.MtlArrayStyle{N}, ::typeof(SpecialFunctions.loggamma), x) where N
    return SpecialFunctions.loggamma.(Broadcast.materialize(x) |> Array) |> MtlArray
end

@compile_workload begin
    Amica.fit(SingleModelAmica, zeros(Float32, 3_000, 24), m=3, maxiter=1, newt_start_iter=0, show_progress=false, ArrayType=MtlArray)
end

end