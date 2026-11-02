using Amica
using LinearAlgebra
using Random
using Statistics
using Test

@testset "PCA Before ICA" begin
    Random.seed!(1234)

    @testset "full-rank sphering unchanged" begin
        x0 = randn(2000, 6) .* [1.0, 2.0, 0.5, 3.0, 1.5, 2.5]'
        x = copy(x0)
        S, lldet, y = Amica.sphering!(x)
        @test y === x # full rank still updates in place
        @test size(S) == (6, 6)
        @test S ≈ S'
        # sample cov of whitened data is identity up to sampling noise
        # (sphering whitens with x'x/N, cov() is Bessel-corrected)
        @test cov(y) ≈ Matrix(1.0I, 6, 6) atol = 0.05
        @test lldet ≈ logabsdet(S)[1]

        # ncomps == nchan takes the same full-rank path
        x2 = copy(x0)
        S2, lldet2, y2 = Amica.sphering!(x2; ncomps = 6)
        @test S2 ≈ S
        @test lldet2 ≈ lldet
        @test y2 ≈ y

        # ncomps > nchan clamps to full rank
        x3 = copy(x0)
        S3, _, y3 = Amica.sphering!(x3; ncomps = 20)
        @test S3 ≈ S
        @test y3 ≈ y
    end

    @testset "PCA reduction" begin
        # 8 channels driven by 3 well-separated sources
        N = 5000
        Random.seed!(5678)
        t = range(0, 20π; length = N)
        S_true = hcat(sin.(t), rand(N) .- 0.5, -log.(rand(N)) .* sign.(rand(N) .- 0.5))
        S_true .-= mean(S_true; dims = 1)
        A_true = randn(8, 3)
        data = S_true * A_true' # (N, 8), rank 3, zero mean

        # same covariance convention as sphering! (x'x/N, no Bessel correction)
        C = data' * data / N
        λ = sort(eigvals(Symmetric(C)); rev = true)

        x = copy(data)
        Sk, lldet, y = Amica.sphering!(x; ncomps = 3)
        @test y !== x # reduced data is a new matrix
        @test size(Sk) == (8, 3)
        @test size(y) == (N, 3)
        @test cov(y) ≈ Matrix(1.0I, 3, 3) atol = 0.05

        # lldet is the log-Jacobian over the retained eigenvalues
        @test lldet ≈ -sum(log.(λ[1:3])) / 2 atol = 1e-8

        # S is the top eigenvectors scaled by the inverse square-root eigenvalues
        U = eigvecs(Symmetric(C))
        Uk = U[:, end:-1:end-2] # top-3 eigenvectors (eigen is ascending)
        for j = 1:3
            @test abs.(Sk[:, j]) ≈ abs.(Uk[:, j]) ./ sqrt(λ[j]) atol = 1e-8
        end
    end

    @testset "fit with PCA reduction" begin
        N = 5000
        Random.seed!(5678)
        t = range(0, 20π; length = N)
        S_true = hcat(sin.(t), rand(N) .- 0.5, -log.(rand(N)) .* sign.(rand(N) .- 0.5))
        S_true .-= mean(S_true; dims = 1)
        A_true = randn(8, 3)
        data = S_true * A_true'

        model = fit(
            SingleModelAmica,
            data;
            ncomps = 3,
            m = 2,
            maxiter = 300,
            show_progress = false,
        )
        @test size(model.A) == (3, 3)
        @test size(Amica.sphering(model)) == (8, 3)

        u = unmixing(model)
        mx = mixing(model)
        @test size(u) == (8, 3)
        @test size(mx) == (3, 8)

        # back-projection from the original all-channel data
        data_c = data .- mean(data; dims = 1)
        sources = recover_sources(data_c, model)
        @test size(sources) == (N, 3)

        # the retained subspace spans the rank-3 data: reconstruction is exact
        @test sources * mx ≈ data_c atol = 1e-3

        # LLdetS matches the retained whitening log-Jacobian
        C = data_c' * data_c / N
        λ = sort(eigvals(Symmetric(C)); rev = true)
        @test model.LLdetS ≈ -sum(log.(λ[1:3])) / 2 atol = 1e-8

        # sources recovered up to sign and permutation
        corr = abs.(cor(sources, S_true))
        @test all(maximum(corr; dims = 1) .> 0.9)
        @test all(maximum(corr; dims = 2) .> 0.9)
    end

    @testset "full-rank fit still works" begin
        data = randn(2000, 6)
        model = fit(SingleModelAmica, data; m = 2, maxiter = 5, show_progress = false)
        @test size(model.A) == (6, 6)
        @test size(unmixing(model)) == (6, 6)
        @test size(mixing(model)) == (6, 6)
        @test size(recover_sources(data, model)) == (2000, 6)
    end
end
