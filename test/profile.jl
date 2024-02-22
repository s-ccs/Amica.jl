using Amica
using MAT
using Profile
using Test

file = matopen("test/testdata/eeg_data.mat")
#file = matopen("test/testdata/pink_sinus_data.mat")

x = read(file, "x")

beta_init = read(file, "beta_init")
mu_init = read(file, "mu_init")
A_init = read(file, "A_init")

close(file)
# before: ca 1.2s / iter
# last best: 0,53s / iter

# SingleModelAmica{Float64, 32, 3} with:
#     - signal-size: (32, 59850)
#     - likelihood: -0.7354624215419704 (after 30 iterations) 

@profview myAmica = fit(SingleModelAmica, x; maxiter=30, do_sphering=true, remove_mean=true, m=3, scale=beta_init[:, :, 1], location=mu_init[:, :, 1], A=copy(A_init[:, :, 1]))

# x = [1 4; 4 1]*Float64.([1.0 2 3; 4 5 6])
# A_init = [1.0 0.003; -0.05 1.0]
# beta_init = [1.1 0.9; 1.0 0.9; 0.9 0.8]
# mu_init = [0.1 0.9; -0.01 0.0; 0.0 -0.02]
# am = fit(SingleModelAmica,x;maxiter=6, m=3, scale=beta_init, location=mu_init, A=copy(A_init), do_sphering = false)
# @test am.A ≈ [0.8761290481633254 0.7147631091971822; 0.48207664428431446 0.6993666404188701]
# @test am.LL[6] ≈ -1.701977346216155
# @test am.Lt[3] ≈ -3.4842563175935526

