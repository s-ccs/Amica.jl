#generate Input. TODO: Use GaussianMixtures.jl instead
function generateInput()
	n_g = 2
	N_g = 10000
	mu_g = [0, 0, 0]
	beta_g = [1/3, 3, 30]
	rho_g = [100, 100, 100]
	
	x = zeros(n_g,N_g)
	for i in 1:n_g
		x[i,:] = mu_g[i] .+ (1/sqrt(beta_g[i])) * rand(Gamma(1/rho_g[i],1),1,N_g).^(1/rho_g[i]).* (((rand(1,N_g).<0.5).*2).-1)
	end
	return x
end