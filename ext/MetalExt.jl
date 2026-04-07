module MetalExt

using Metal, LinearAlgebra, SpecialFunctions, Statistics, Amica

import Base.Broadcast

_to_metal(x::AbstractArray{T}) where {T} = MtlArray{T}(x)

function LinearAlgebra.:\(a::MtlArray{T}, b::MtlArray{T}) where {T}
    return _to_metal((a |> Array{T}) \ (b |> Array{T}))
end

function LinearAlgebra.inv(a::MtlArray{T}) where {T}
    return _to_metal(LinearAlgebra.inv(a |> Array{T}))
end

function Broadcast.broadcasted(
    ::Metal.MtlArrayStyle{N},
    ::typeof(SpecialFunctions.loggamma),
    x,
) where {N}
    return _to_metal(SpecialFunctions.loggamma.(Broadcast.materialize(x) |> Array))
end


function LinearAlgebra.svd(x::MtlArray{T,2}; kwargs...) where {T}
    return LinearAlgebra.svd(Array(x); kwargs...)
end

function Base.:*(A::MtlArray{T,2}, B::Matrix{T}) where {T}
    return A * _to_metal(B)
end

function LinearAlgebra.mul!(
    C::MtlArray{T,2},
    A::SubArray{T,2,<:MtlArray{T,2}},
    B::Adjoint{T,<:MtlArray{T,2}},
) where {T}
    copyto!(C, _to_metal(Array(A) * Array(B)))
    return C
end

function Base.:*(A::Adjoint{T,<:MtlArray{T,2}}, B::MtlArray{T,2}) where {T}
    return _to_metal(Array(A) * Array(B))
end

end
