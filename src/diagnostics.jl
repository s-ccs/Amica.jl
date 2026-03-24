"Write a Julia array to a binary file in column-major (Fortran-compatible) format"
function write_binary(filepath::String, arr::AbstractArray{T}) where {T}
    open(filepath, "w") do io
        write(io, Array(arr))
    end
end
