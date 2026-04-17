using Test
using Statistics
using LinearAlgebra
using Amica
using Metal

include("util.jl")

build_fortran()

fortran_full_run = integration_test_path("dumps", "fortran", "full_run")
fortran_output_dir = integration_test_path("amicaout")

cleanup_integration_dumps!()
mkpath(fortran_full_run)

amica_exe = integration_test_path("fortran", "amica")
config_path = integration_input_path("amicadefs_fullrun.params")
run(
    setenv(
        Cmd(`$amica_exe $config_path`; dir = integration_test_dir()),
        "HOME" => homedir(),
        "OUT_PATH" => fortran_full_run,
        "OMPI_MCA_plm" => "isolated",
    ),
)

data = read_fdt(
    integration_test_path("input", "small.fdt");
    ncols = 19,
    T = Float32,
    transpose = true,
    OutType = Float32,
)
data_before = copy(data)
metal_data = MtlArray(data)
(N, n) = size(data)

A_init = Float32.(read_fdt(joinpath(fortran_full_run, "A.bin"); ncols = n, T = Float64))
sbeta_init = Float32.(
    read_fdt(
        joinpath(fortran_full_run, "sbeta.bin");
        ncols = 3,
        T = Float64,
        transpose = true,
    ),
)
mu_init = Float32.(
    read_fdt(
        joinpath(fortran_full_run, "mu.bin");
        ncols = 3,
        T = Float64,
        transpose = true,
    ),
)

myAmica = SingleModelAmica(
    metal_data,
    ncomps = n,
    nsamples = N,
    m = 3,
    A = MtlArray(A_init),
    scale = MtlArray(sbeta_init),
    location = MtlArray(mu_init),
    block_size = N,
    num_threads = 1,
)

lrate = Amica.LearningRate{Float32}(
    0.1f0,
    0.05f0;
    shapelratefact = 0.5f0,
    min = 1.0f-8,
    newtrate = 1.0f0,
    newt_ramp = 10,
    minrho = 1.0f0,
    maxrho = 2.0f0,
)

Amica.amica!(
    myAmica,
    metal_data;
    maxiter = 40,
    lrate = lrate,
    remove_mean = true,
    do_sphering = true,
    show_progress = true,
    do_newton = true,
    newt_start_iter = 50,
    iterwin = 1,
    data_inplace = false,
    mindll = 1.0f-9,
)

fortran_LL =
    Float32.(vec(read_fdt(joinpath(fortran_output_dir, "LL"); ncols = 1, T = Float64)))
fortran_A = Float32.(read_fdt(joinpath(fortran_output_dir, "A"); ncols = n, T = Float64))

metal_A = Array(myAmica.A)
metal_LL = Float32.(myAmica.LL)

println("max |A_metal - A_fortran| = ", maximum(abs.(metal_A .- fortran_A)))
println("max |LL_metal - LL_fortran| = ", maximum(abs.(metal_LL .- fortran_LL)))

@test isapprox(metal_A, fortran_A; atol = 5.0f-3, rtol = 0.0f0)
@test isapprox(metal_LL, fortran_LL; atol = 5.0f-3, rtol = 0.0f0)
@test data ≈ data_before
