using MAT
using Amica
using SignalAnalysis
using LinearAlgebra

t = range(0,20*π,length=1000)
s =rand(PinkGaussian(length(t)),4)'
s[2,:] = sin.(t)
s[3,:] = sin.(2 .* t)
s[4,:] = sin.(10 .* t)
s = s .* [1,2,3,4]
#A = rand(size(s,1),size(s,1))
A = [1 1 0 0; 0 1 1 0; 0 0 1 1; 1 0 1 0]

x = A*s
m = 3
M = 2
n = 4

#initialise random parameters before saving them
beta = ones(m, n, M) + 0.1 * randn(m, n, M)

if m > 1
    mu = 0.1 * randn(m, n, M)
else
    mu = zeros(m, n, M)
end

eye = Matrix{Float64}(I, n, n)
A_init = zeros(n,n,M)
for h in 1:M
    A_init[:,:,h] = eye[n] .+ 0.1*rand(n,n)
    for i in 1:n
        A_init[:,i,h] = A_init[:,i,h] / norm(A_init[:,i,h])
    end
end

file = matopen("test/pink_sinus_data.mat", "w")
write(file, "x", x)
write(file, "s", s)
write(file, "A", A)

write(file, "beta_init", beta)
write(file, "A_init", A_init)
write(file, "mu_init", mu)
close(file)

# file = matopen("pink_sinus_data.mat")
# x = read(file, "data")
# close(file)