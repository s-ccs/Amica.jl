using Amica
using Test

@testset "Amica.jl" begin
    A = [1 2; 3 4]
    @test calculate_ldet(A) == -0.693147180559945
end
