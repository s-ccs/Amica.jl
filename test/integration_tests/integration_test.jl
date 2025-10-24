using Test
using Statistics
using Amica

include("util.jl")

@testset "compare against fortran" begin

    build_fortran()
    run_fortran("amicadefs.params")

    # verify the raw data is identical
    data = Float64.(read_fdt("input/Memorize.fdt"; ncols=71, T=Float32))
    raw = read_fdt("datadumps/raw_data_seg_1.bin"; ncols=71, T=Float64)
    @test raw ≈ data

    Amica.removeMean!(data)

    without_mean = read_fdt("datadumps/mean_data_seg_1.bin"; ncols=71, T=Float64)

    @test without_mean ≈ data

    sphered = read_fdt("datadumps/sphere_data_seg_1.bin"; ncols=71, T=Float64)

    Amica.sphering!(data)

    @test sphered ≈ data

    A = read_fdt("datadumps/A.bin"; ncols=71, T=Float64)
    W = read_fdt("datadumps/W.bin"; ncols=71, T=Float64)

    sbeta = read_fdt("datadumps/sbeta.bin"; ncols=3, T=Float64)
    rho = read_fdt("datadumps/rho.bin"; ncols=3, T=Float64)
    mu = read_fdt("datadumps/mu.bin"; ncols=3, T=Float64)

    data = Float64.(read_fdt("input/Memorize.fdt"; ncols=71, T=Float32))

    myAmica = SingleModelAmica(data; maxiter=30, do_sphering=true, remove_mean=true, m=3, A=A, scale=sbeta, location=mu)

    @test A ≈ inv(W)


    lrate = Amica.LearningRate{Float64}()
    # run amica for one iteration
    Amica.amica!(myAmica, data, maxiter=1, lrate=lrate)

    # test calculate_ldet!
    Dsum = read_fdt("datadumps/Dsum.bin"; ncols=1, T=Float64)
    @test Dsum[1, 1] ≈ myAmica.ldet

    # test update_sources!
    b = read_fdt("datadumps/b.bin"; ncols=639000, T=Float64)'[:, 1:319500]
    @test b ≈ myAmica.source_signals

    # test calculate_y!
    y = read_3d_fdt("datadumps/y.bin"; ncols=639000, nslabs=3, T=Float64)
    y = permutedims(y, (3, 2, 1))[:, :, 1:319500]

    @test y ≈ myAmica.y

    @info size(myAmica.z)

    # test calculate_u!
    z = read_3d_fdt("datadumps/z.bin"; ncols=639000, nslabs=3, T=Float64)
    z = permutedims(z, (3, 2, 1))[:, :, 1:319500]
    @test z ≈ myAmica.z

    # test calculate_Lt!
    Ptmp = read_fdt("datadumps/Ptmp.bin"; ncols=639000, T=Float64)[1:319500, 1]
    @test Ptmp ≈ myAmica.Lt


    # test calculate_LL!
    LL = read_fdt("datadumps/LL.bin"; ncols=1, T=Float64)[1, :]

    @test LL[1] ≈ myAmica.LL[1]

    @info lrate
    @info lrate.lrate

end
