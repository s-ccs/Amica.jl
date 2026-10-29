using Test
using LinearAlgebra
using Random
using SpecialFunctions
using Amica

# squared variance of each component's contribution to the raw (pre-sphering)
# data, recomputed from the model. Same formula as sort_according_to_var!:
# model source variance (GGD mixture) times squared norm of the sensor-space
# mixing map (row of Amica.mixing()).
function comp_raw_var(amica)
    n = size(amica.A, 2)
    mix = Amica.mixing(amica)
    map_norms = dropdims(sum(mix .^ 2, dims = 2), dims = 2)
    p = Matrix(amica.proportions)
    mu = Matrix(amica.location)
    b = Matrix(amica.scale)
    r = Matrix(amica.shape)
    v = zeros(n)
    for i in 1:n
        second = (gamma.(3 ./ r[i, :]) ./ gamma.(1 ./ r[i, :])) ./ b[i, :] .^ 2
        v[i] = sum(p[i, :] .* (mu[i, :] .^ 2 .+ second)) * map_norms[i]
    end
    return v
end

@testset "sort_by_variance" begin
    Random.seed!(42)
    N, n, m = 3_000, 5, 3
    # sources with very different variances (spiky and gaussian), randomly mixed
    S = hcat(
        10 * (rand(N) .< 0.15) .* randn(N),
        randn(N),
        0.2 * (rand(N) .< 0.5) .* randn(N),
        0.5 * randn(N),
        0.1 * randn(N),
    )
    M = randn(n, n)
    x = S * M'

    amica = Amica.fit(
        SingleModelAmica,
        x;
        m = m,
        maxiter = 100,
        show_progress = false,
        block_size = N,
    )
    ref = deepcopy(amica)

    Amica.sort_according_to_var!(amica)

    # sorting permutes the components: every sorted source equals an unsorted one
    s_ref = Amica.recover_sources(x, ref)
    s_sorted = Amica.recover_sources(x, amica)
    for i in 1:n
        d = minimum(norm(s_sorted[:, i] .- s_ref[:, j]) for j in 1:n)
        @test d < 1e-8 * norm(s_sorted[:, i])
    end

    # the sorted model's component variances are in descending order
    v = comp_raw_var(amica)
    @test all(diff(v) .<= 1e-9 * maximum(v))

    # and through the public API: sort_by_variance=true sorts in-place
    amica2 = Amica.fit(
        SingleModelAmica,
        x;
        m = m,
        maxiter = 100,
        show_progress = false,
        block_size = N,
        sort_by_variance = true,
    )
    v2 = comp_raw_var(amica2)
    @test all(diff(v2) .<= 1e-9 * maximum(v2))
end
