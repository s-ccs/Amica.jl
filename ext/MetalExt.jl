module MetalExt

using Metal, LinearAlgebra, SpecialFunctions
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

end