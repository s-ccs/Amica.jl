using Test
using Statistics
using Amica

function read_fdt(path::String; ncols::Int, T::Type=Float32)::Array{T,2}
    file_size = Base.filesize(path)
    nvals = file_size ÷ sizeof(T)
    nrows = nvals ÷ ncols
    data = reinterpret(T, read(path))
    return reshape(data, ncols, nrows)
end

@testset "run against memorize set" begin
    # verify the raw data is identical
    data = Float64.(read_fdt("input/Memorize.fdt"; ncols=71, T=Float32))

    myAmica = fit(SingleModelAmica, data; maxiter=30, do_sphering=true, remove_mean=true, m=3)
end
