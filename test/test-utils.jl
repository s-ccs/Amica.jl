using LinearAlgebra

@testset "mixing / unmixing / sphering" begin
    N = 7
    n = 3

    A = [
        1.2 0.1 0.0
        0.2 0.9 0.3
        0.0 0.2 1.1
    ]
    model = SingleModelAmica(Float64, ncomps = n, nsamples = N, m = 2, A = A)

    S = [
        1.5 0.0 0.0
        0.1 1.3 0.0
        0.0 0.2 0.8
    ]
    model.S .= S

    W_expected = S * inv(A)'
    M_expected = A' * inv(S)

    @test Amica.unmixing(model) ≈ W_expected
    @test Amica.mixing(model) ≈ M_expected

    # Unmixing and mixing should be inverses in sample-by-channel convention.
    @test Amica.unmixing(model) * Amica.mixing(model) ≈ Matrix(I, n, n)
    @test Amica.mixing(model) * Amica.unmixing(model) ≈ Matrix(I, n, n)

    data = reshape(collect(1.0:(N*n)), N, n)
    sources = Amica.recover_sources(data, model)

    @test sources ≈ data * W_expected
    @test sources * Amica.mixing(model) ≈ data
end
