using Amica
using Test

@testset "Amica.jl" begin
    @test Amica.calculate_y(2.132083330492723,-0.16800241654969805,[1.7759370173937195, -0.5593467100030134, -0.12437952916769522, -0.3875743344924953, -1.0946943240448705]) == [2.83847332422518, -0.5714274416984326, 0.06369663580943559, -0.3206113425896638, -1.3531235661311032]
    x = [1 4; 4 1]*Float64.([1.0 2 3; 4 5 6])
    A_init = [1.0 0.003; -0.05 1.0]
    beta_init = [1.1 0.9; 1.0 0.9; 0.9 0.8]
    mu_init = [0.1 0.9; -0.01 0.0; 0.0 -0.02]
    am = fit(SingleModelAmica,x;maxiter=50,M=1, m=3, beta=beta_init, mu=mu_init, A=copy(A_init))
    @test am.A == [0.9919369640765601 0.4570871372821922; 0.12673223464682265 0.8894219184004689]
end
