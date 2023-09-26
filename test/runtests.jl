using Amica
using Test
using MAT

#Needs supersmall_data.mat and supersmall_data_results.mat
@testset "Amica.jl" begin
    #Test single model
    x = [1 4; 4 1]*Float64.([1.0 2 3; 4 5 6])
    A_init = [1.0 0.003; -0.05 1.0]
    beta_init = [1.1 0.9; 1.0 0.9; 0.9 0.8]
    mu_init = [0.1 0.9; -0.01 0.0; 0.0 -0.02]
    am = fit(SingleModelAmica,x;maxiter=6, m=3, scale=beta_init, location=mu_init, A=copy(A_init), do_sphering = false)
    @test am.A == [0.8761290481633254 0.7147631091971822; 0.48207664428431446 0.6993666404188701]
    @test am.LL[6] == -1.701977346216155
    @test am.Lt[3] == -3.4842563175935526

    #Test multi model
    file = matopen("test/testdata/supersmall_data.mat")
    beta_init = read(file, "beta_init")
    mu_init = read(file, "mu_init")
    A_init = read(file, "A_init")
    x = read(file, "x")
    close(file)
    file = matopen("test/testdata/supersmall_data_results.mat") #contains results for A and LL after 4 iterations (2 Models, 3 GGs)
    am = fit(MultiModelAmica,x;maxiter=4, m=3,M = 2,scale=beta_init, location=mu_init, A=copy(A_init), do_sphering = false)
    @test am.LL == read(file,"LL_after_4iter")
    @test am.models[1].A == read(file, "A1_after_4iter")
    @test am.models[2].A == read(file, "A2_after_4iter")
    close(file)
end
