using Amica
using Test
using MAT

#Needs supersmall_data.mat and supersmall_data_results.mat
@testset "Amica.jl" begin
    #@test Amica.calculate_y(2.132083330492723,-0.16800241654969805,[1.7759370173937195, -0.5593467100030134, -0.12437952916769522, -0.3875743344924953, -1.0946943240448705]) == [2.83847332422518, -0.5714274416984326, 0.06369663580943559, -0.3206113425896638, -1.3531235661311032]
    
    #Test single model
    x = [1 4; 4 1]*Float64.([1.0 2 3; 4 5 6])
    A_init = [1.0 0.003; -0.05 1.0]
    beta_init = [1.1 0.9; 1.0 0.9; 0.9 0.8]
    mu_init = [0.1 0.9; -0.01 0.0; 0.0 -0.02]
    am = fit(SingleModelAmica,x;maxiter=6, m=3, beta=beta_init, mu=mu_init, A=copy(A_init))
    @test am.A == [0.8761290481633254 0.7147631091971822; 0.48207664428431446 0.6993666404188701]
    @test am.LL[6] == -1.701977346216155
    @test am.Lt[3] == -3.4842563175935526

    #Test multi model
    file = matopen("test/supersmall_data.mat")
    beta_init = read(file, "beta_init")
    mu_init = read(file, "mu_init")
    A_init = read(file, "A_init")
    x = read(file, "x")
    close(file)
    file = matopen("test/supersmall_data_results.mat") #contains results for A and LL after 4 iterations (2 Models, 3 GGs)
    am = fit(MultiModelAmica,x;maxiter=4, m=3,M = 2,beta=beta_init, mu=mu_init, A=copy(A_init))
    @test am.LL == read(file,"LL_after_4iter")
    @test am.models[1].A == read(file, "A1_after_4iter")
    @test am.models[2].A == read(file, "A2_after_4iter")
    close(file)
end
