using Test
using Statistics
using Amica
using LinearAlgebra

include("util.jl")

build_fortran()

cleanup_integration_dumps!()
prepare_integration_dump_dirs!()

fortran_without_newton = integration_dump_dir(:fortran, :without_newton)
fortran_with_newton = integration_dump_dir(:fortran, :with_newton)
julia_without_newton = integration_dump_dir(:julia, :without_newton)
julia_with_newton = integration_dump_dir(:julia, :with_newton)
fortran_output_dir = integration_test_path("amicaout")

try
    run_fortran("amicadefs.params", fortran_without_newton)

    @testset "basic tests" begin
        data = read_fdt(integration_test_path("input", "small.fdt"); ncols=19, T=Float32, transpose=true, OutType=Float64)
        raw = read_fdt(joinpath(fortran_without_newton, "raw_data_seg_1.bin"); ncols=19, T=Float64, transpose=true)
        @test raw ≈ data

        Amica.removeMean!(data)

        without_mean = read_fdt(joinpath(fortran_without_newton, "mean_data_seg_1.bin"); ncols=19, T=Float64, transpose=true)
        @test without_mean ≈ data

        sphered = read_fdt(joinpath(fortran_without_newton, "sphere_data_seg_1.bin"); ncols=19, T=Float64, transpose=true)

        Amica.sphering!(data)
        @test sphered ≈ data
    end


    @testset "compare without newton" begin
        A = read_fdt(joinpath(fortran_without_newton, "A.bin"); ncols=19, T=Float64)
        W = read_fdt(joinpath(fortran_without_newton, "W.bin"); ncols=19, T=Float64)

        sbeta = read_fdt(joinpath(fortran_without_newton, "sbeta.bin"); ncols=3, T=Float64, transpose=true)
        rho = read_fdt(joinpath(fortran_without_newton, "rho.bin"); ncols=3, T=Float64, transpose=true)
        mu = read_fdt(joinpath(fortran_without_newton, "mu.bin"); ncols=3, T=Float64, transpose=true)

        data = read_fdt(integration_test_path("input", "small.fdt"); ncols=19, T=Float32, transpose=true, OutType=Float64)
        data_before = copy(data)

        (N, n) = size(data)

        myAmica = SingleModelAmica(Float64, ncomps=n, nsamples=N, m=3, A=A, scale=sbeta, location=mu, block_size=N)

        lrate = Amica.LearningRate{Float64}()
        Amica.amica!(myAmica, data, maxiter=1, lrate=lrate, newt_start_iter=50, data_inplace=false; dump_dir=julia_without_newton)

        b_fortran = read_fdt(joinpath(fortran_without_newton, "b.bin"); ncols=2 * N, T=Float64, transpose=true)[:, 1:N]'
        b_julia = read_fdt(joinpath(julia_without_newton, "source_signals.bin"); ncols=N, T=Float64)
        @test b_fortran ≈ b_julia

        g_fortran = read_fdt(joinpath(fortran_without_newton, "g_after_iter1.bin"); ncols=2 * N, T=Float64, transpose=true)[:, 1:N]'
        g_julia = read_fdt(joinpath(julia_without_newton, "g.bin"); ncols=N, T=Float64)
        @test g_fortran ≈ g_julia

        y_fortran = read_3d_fdt(joinpath(fortran_without_newton, "y.bin"); ncols=2 * N, nslabs=3, T=Float64)[1:N, :, :]
        y_julia = read_3d_fdt(joinpath(julia_without_newton, "y.bin"); ncols=N, nslabs=3, T=Float64)
        @test y_fortran ≈ y_julia

        z_fortran = read_3d_fdt(joinpath(fortran_without_newton, "z.bin"); ncols=2 * N, nslabs=3, T=Float64)[1:N, :, :]
        z_julia = read_3d_fdt(joinpath(julia_without_newton, "z.bin"); ncols=N, nslabs=3, T=Float64)
        @test z_fortran ≈ z_julia

        Ptmp = read_fdt(joinpath(fortran_without_newton, "Ptmp.bin"); ncols=2 * N, T=Float64)[1:N, 1]
        @test Ptmp ≈ myAmica.Lt

        LL = read_fdt(joinpath(fortran_without_newton, "LL.bin"); ncols=1, T=Float64)[1, :]'
        @test LL[1] ≈ myAmica.LL[1]

        mu_1 = read_fdt(joinpath(fortran_without_newton, "mu_1.bin"); ncols=3, T=Float64, transpose=true)
        @test myAmica.location ≈ mu_1

        sbeta_1 = read_fdt(joinpath(fortran_without_newton, "sbeta_1.bin"); ncols=3, T=Float64, transpose=true)
        @test myAmica.scale ≈ sbeta_1

        rho_1 = read_fdt(joinpath(fortran_without_newton, "rho_1.bin"); ncols=3, T=Float64, transpose=true)
        @test myAmica.shape ≈ rho_1

        A_1 = read_fdt(joinpath(fortran_without_newton, "a_after_iter1.bin"); ncols=19, T=Float64)
        @test myAmica.A ≈ A_1


        alpha = read_fdt(joinpath(fortran_without_newton, "alpha_1.bin"); ncols=3, T=Float64, transpose=true)
        @test myAmica.proportions ≈ alpha
        @test data ≈ data_before
    end

    run_fortran("amicadefs_newton.params", fortran_with_newton)

    @testset "compare with newton" begin
        A = read_fdt(joinpath(fortran_with_newton, "A.bin"); ncols=19, T=Float64)
        W = read_fdt(joinpath(fortran_with_newton, "W.bin"); ncols=19, T=Float64)
        @test W ≈ inv(A)


        sbeta = read_fdt(joinpath(fortran_with_newton, "sbeta.bin"); ncols=3, T=Float64, transpose=true)
        rho = read_fdt(joinpath(fortran_with_newton, "rho.bin"); ncols=3, T=Float64, transpose=true)
        mu = read_fdt(joinpath(fortran_with_newton, "mu.bin"); ncols=3, T=Float64, transpose=true)

        data = read_fdt(integration_test_path("input", "small.fdt"); ncols=19, T=Float32, transpose=true, OutType=Float64)
        data_before = copy(data)

        (N, n) = size(data)

        myAmica = SingleModelAmica(Float64, ncomps=n, nsamples=N, m=3, A=A, scale=sbeta, location=mu, block_size=N)

        lrate = Amica.LearningRate{Float64}()
        Amica.amica!(myAmica, data, maxiter=1, lrate=lrate, newt_start_iter=0, data_inplace=false; dump_dir=julia_with_newton)

        b_fortran = read_fdt(joinpath(fortran_with_newton, "b.bin"); ncols=2 * N, T=Float64, transpose=true)[:, 1:N]'
        b_julia = read_fdt(joinpath(julia_with_newton, "source_signals.bin"); ncols=N, T=Float64)
        @test b_fortran ≈ b_julia

        g_fortran = read_fdt(joinpath(fortran_with_newton, "g_after_iter1.bin"); ncols=2 * N, T=Float64, transpose=true)[:, 1:N]'
        g_julia = read_fdt(joinpath(julia_with_newton, "g.bin"); ncols=N, T=Float64)
        @test g_fortran ≈ g_julia

        y_fortran = read_3d_fdt(joinpath(fortran_with_newton, "y.bin"); ncols=2 * N, nslabs=3, T=Float64)[1:N, :, :]
        y_julia = read_3d_fdt(joinpath(julia_with_newton, "y.bin"); ncols=N, nslabs=3, T=Float64)
        @test y_fortran ≈ y_julia

        z_fortran = read_3d_fdt(joinpath(fortran_with_newton, "z.bin"); ncols=2 * N, nslabs=3, T=Float64)[1:N, :, :]
        z_julia = read_3d_fdt(joinpath(julia_with_newton, "z.bin"); ncols=N, nslabs=3, T=Float64)
        @test z_fortran ≈ z_julia

        Ptmp = read_fdt(joinpath(fortran_with_newton, "Ptmp.bin"); ncols=2 * N, T=Float64)[1:N, 1]
        @test Ptmp ≈ myAmica.Lt

        LL = read_fdt(joinpath(fortran_with_newton, "LL.bin"); ncols=1, T=Float64)[1, :]'
        @test LL[1] ≈ myAmica.LL[1]

        mu_1 = read_fdt(joinpath(fortran_with_newton, "mu_1.bin"); ncols=3, T=Float64, transpose=true)
        @test myAmica.location ≈ mu_1

        sbeta_1 = read_fdt(joinpath(fortran_with_newton, "sbeta_1.bin"); ncols=3, T=Float64, transpose=true)
        @test myAmica.scale ≈ sbeta_1

        rho_1 = read_fdt(joinpath(fortran_with_newton, "rho_1.bin"); ncols=3, T=Float64, transpose=true)
        @test myAmica.shape ≈ rho_1

        A_1 = read_fdt(joinpath(fortran_with_newton, "a_after_iter1.bin"); ncols=19, T=Float64)
        @test myAmica.A ≈ A_1

        kappa = read_fdt(joinpath(fortran_with_newton, "kappa_after_iter1.bin"); ncols=19, T=Float64)
        @test myAmica.newton_kappa ≈ kappa

        lambda = read_fdt(joinpath(fortran_with_newton, "lambda_after_iter1.bin"); ncols=19, T=Float64)
        @test myAmica.newton_lambda ≈ lambda
        sigma2 = read_fdt(joinpath(fortran_with_newton, "sigma2_after_iter1.bin"); ncols=19, T=Float64)
        @test myAmica.newton_sigma2 ≈ sigma2[:, 1]

        alpha = read_fdt(joinpath(fortran_with_newton, "alpha_1.bin"); ncols=3, T=Float64, transpose=true)
        @test myAmica.proportions ≈ alpha
        @test data ≈ data_before
    end

    @testset "full run" begin
        fortran_full_run = integration_test_path("dumps", "fortran", "full_run")

        run_fortran("amicadefs_fullrun.params", fortran_full_run)

        data = read_fdt(integration_test_path("input", "small.fdt"); ncols=19, T=Float32, transpose=true, OutType=Float64)
        data_before = copy(data)
        (N, n) = size(data)

        A_init = read_fdt(joinpath(fortran_full_run, "A.bin"); ncols=n, T=Float64)
        sbeta_init = read_fdt(joinpath(fortran_full_run, "sbeta.bin"); ncols=3, T=Float64, transpose=true)
        mu_init = read_fdt(joinpath(fortran_full_run, "mu.bin"); ncols=3, T=Float64, transpose=true)

        myAmica = SingleModelAmica(
            Float64,
            ncomps=n,
            nsamples=N,
            m=3,
            A=A_init,
            scale=sbeta_init,
            location=mu_init,
            block_size=N,
            num_threads=1,
        )

        lrate = Amica.LearningRate{Float64}(
            0.1,
            0.05;
            shapelratefact=0.5,
            min=1.0e-8,
            newtrate=1.0,
            newt_ramp=10,
            minrho=1.0,
            maxrho=2.0,
        )

        Amica.amica!(
            myAmica,
            data;
            maxiter=40,
            lrate=lrate,
            remove_mean=true,
            do_sphering=true,
            show_progress=false,
            do_newton=true,
            newt_start_iter=50,
            iterwin=1,
            data_inplace=false,
            mindll=1.0e-9,
        )

        fortran_LL = vec(read_fdt(joinpath(fortran_output_dir, "LL"); ncols=1, T=Float64))

        fortran_A = read_fdt(joinpath(fortran_output_dir, "A"); ncols=n, T=Float64)
        @test isapprox(myAmica.A, fortran_A; atol=1e-3, rtol=0)
        @test isapprox(myAmica.LL, fortran_LL; atol=1e-3, rtol=0)
        @test data ≈ data_before
    end
finally
    cleanup_integration_dumps!()
end
