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

@testset "compare against fortran" begin
    # verify the raw data is identical
    data = Float64.(read_fdt("Memorize.fdt"; ncols=71, T=Float32))
    raw = read_fdt("fortran_data/raw_data_seg_1.bin"; ncols=71, T=Float64)
    @test raw ≈ data

    # @info "first three" raw[1:1, 1:3] size(raw)

    Amica.removeMean!(data)

    # compare the data after removing the means
    without_mean = read_fdt("fortran_data/mean_data_seg_1.bin"; ncols=71, T=Float64)

    @test without_mean ≈ data

    sphered = read_fdt("fortran_data/sphere_data_seg_1.bin"; ncols=71, T=Float64)

    Amica.sphering!(data)

    @test sphered ≈ data

    
end
