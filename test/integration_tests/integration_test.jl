using Test
using Statistics
using Amica
using LinearAlgebra

include("util.jl")

@testset "compare against fortran" begin

    # build_fortran()
    # run_fortran("amicadefs.params")

    # verify the raw data is identical
    data = Float64.(read_fdt("input/Memorize.fdt"; ncols=71, T=Float32))'
    raw = read_fdt("datadumps/raw_data_seg_1.bin"; ncols=71, T=Float64)'
    @test raw ≈ data

    Amica.removeMean!(data)

    without_mean = read_fdt("datadumps/mean_data_seg_1.bin"; ncols=71, T=Float64)'
    @test without_mean ≈ data

    sphered = read_fdt("datadumps/sphere_data_seg_1.bin"; ncols=71, T=Float64)'

    Amica.sphering!(data)
    @test sphered ≈ data

    A = read_fdt("datadumps/A.bin"; ncols=71, T=Float64)
    A_init = read_fdt("datadumps/A_init.bin"; ncols=71, T=Float64)
    W = read_fdt("datadumps/W.bin"; ncols=71, T=Float64)


    sbeta = read_fdt("datadumps/sbeta.bin"; ncols=3, T=Float64)'
    rho = read_fdt("datadumps/rho.bin"; ncols=3, T=Float64)'
    mu = read_fdt("datadumps/mu.bin"; ncols=3, T=Float64)'

    data = Float64.(read_fdt("input/Memorize.fdt"; ncols=71, T=Float32))'

    (N, n) = size(data)


    myAmica = SingleModelAmica(Float64, ncomps=n, nsamples=N, m=3, A=A, scale=sbeta, location=mu)

    @test A ≈ inv(W)

    lrate = Amica.LearningRate{Float64}()
    # run amica for one iteration
    Amica.amica!(myAmica, data, maxiter=1, lrate=lrate, newt_start_iter=0)

    # test update_sources!
    b = read_fdt("datadumps/b.bin"; ncols=639000, T=Float64)'[:, 1:319500]'
    dev = mean(abs.(b .- myAmica.source_signals))
    @test b ≈ myAmica.source_signals
    println("Source signals deviation: $dev")

    # test calculate_y!
    y = read_3d_fdt("datadumps/y.bin"; ncols=639000, nslabs=3, T=Float64)[1:319500, :, :]
    dev = mean(abs.(y .- myAmica.y))
    @test y ≈ myAmica.y
    println("y deviation: $dev")

    # test calculate_u!
    z = read_3d_fdt("datadumps/z.bin"; ncols=639000, nslabs=3, T=Float64)[1:319500, :, :]
    dev = mean(abs.(z .- myAmica.z))
    @test z ≈ myAmica.z
    println("z deviation: $dev")

    # test calculate_Lt!
    Ptmp = read_fdt("datadumps/Ptmp.bin"; ncols=639000, T=Float64)[1:319500, 1]
    dev = mean(abs.(Ptmp .- myAmica.Lt))
    @test Ptmp ≈ myAmica.Lt
    println("Lt deviation: $dev")

    # test calculate_LL!
    LL = read_fdt("datadumps/LL.bin"; ncols=1, T=Float64)[1, :]'
    dev = abs(LL[1] - myAmica.LL[1])
    @test LL[1] ≈ myAmica.LL[1]
    println("LL deviation: $dev")

    # test g 
    g = read_fdt("datadumps/g_after_iter1.bin"; ncols=639000, T=Float64)[1:319500, :]
    dev = mean(abs.(g .- myAmica.g))
    @test g ≈ myAmica.g
    println("g deviation: $dev")

    # location after one iteration
    mu_1 = read_fdt("datadumps/mu_1.bin"; ncols=3, T=Float64)'
    dev = mean(abs.(myAmica.location .- mu_1))
    @test myAmica.location ≈ mu_1
    println("Location deviation: $dev")

    # scale after one iteration
    sbeta_1 = read_fdt("datadumps/sbeta_1.bin"; ncols=3, T=Float64)'
    dev = mean(abs.(myAmica.scale .- sbeta_1))
    @test myAmica.scale ≈ sbeta_1
    println("Scale deviation: $dev")

    # shape
    rho_1 = read_fdt("datadumps/rho_1.bin"; ncols=3, T=Float64)'
    dev = mean(abs.(myAmica.shape .- rho_1))
    @test myAmica.shape ≈ rho_1
    println("Shape deviation: $dev")

    # A
    A_1 = read_fdt("datadumps/a_after_iter1.bin"; ncols=71, T=Float64)
    dev = mean(abs.(myAmica.A .- A_1))
    @test myAmica.A ≈ A_1
    println("A deviation: $dev")

    # kappa
    kappa = read_fdt("datadumps/kappa_after_iter1.bin"; ncols=71, T=Float64)
    dev = mean(abs.(myAmica.newton_kappa .- kappa))
    @test myAmica.newton_kappa ≈ kappa
    println("Kappa deviation: $dev")
    # lambda
    lambda = read_fdt("datadumps/lambda_after_iter1.bin"; ncols=71, T=Float64)
    dev = mean(abs.(myAmica.newton_lambda .- lambda))
    @test myAmica.newton_lambda ≈ lambda
    println("Lambda deviation: $dev")

    alpha = read_fdt("datadumps/alpha_1.bin"; ncols=3, T=Float64)'
    dev = mean(abs.(myAmica.proportions .- alpha))
    @test myAmica.proportions ≈ alpha
    println("Alpha deviation: $dev")
end
