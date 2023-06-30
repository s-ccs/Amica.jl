using Distributions

# Based on https://github.com/sccn/amica/wiki/gg6.m
# Generate N Generalized Gaussian m-dimensional vectors with means mu(:),
# inverse scales beta(:), and parameters rho(:).
function gg6(n, N, mu, beta, rho)
    x = zeros(n,N)
    for i in 1:n
        x[i,:] = mu[i] .+ (1/sqrt(beta[i])) * rand(Gamma(1/rho[i],1),1,N_g).^(1/rho[i]).* (((rand(1,N).<0.5).*2).-1)
    end
    return x
end
