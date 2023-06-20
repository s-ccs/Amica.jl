"""
Main AMICA algorithm

"""
function amica(x, M, m, maxiter, update_rho, mindll, iterwin, do_newton, remove_mean)
	#As
	#cs
	#Variables
	(n, N) = size(x)
	lrate0 = 0.1
	lratemax = 1.0
	lnatrate = 0.1
	newt_start_iter = 1 #eig. 25
	lratefact = 0.5

	lrate = lrate0

	dispinterval = 50
	
	showLL = 1
	plotfig = 1

	Mx = maximum(abs.(x)) #maximum and max are not the same

	rholrate = 0.1
	rhomin = 1.0/2
	rhomax = 5
	rho0 = 1.5 #usually depends on update_rho

	plothist = 1
	nbins = 50

	mn = mean(x, dims = 2) #should be zeros if remove_mean = 0

	if remove_mean == 1
		removeMean(x)
	end
	#initialize parameters
	A = zeros(n,n,M)
	centers = zeros(n,M)
	eye = Matrix{Float64}(I, n, n)

	for h in 1:M #todo: wieder randomisieren
		A[:,:,h] = eye[n] .+ 0.1*rand(n,n)
		for i in 1:n
			A[:,i,h] = A[:,i,h] / norm(A[:,i,h])
		end
		A[:,:,1] = [1.0 0.003; -0.05 1.0]
		A[:,:,2] = [2.0 0.003; -0.05 1.0]
		#A = [1.0 0.003; -0.05 1.0]
		centers[:,h] = zeros(n,1)
	end

	proportions = (1/M) * ones(M,1)
	alpha = (1/m) * ones(m,n,M)
	
	if m > 1
		#mu = [0.1 0.9; -0.01 0.0; 0.0 -0.02]
		mu = 0.1 * randn(m, n, M)
		mu[:,:,1] = [0.1 0.9; -0.01 0.0; 0.0 -0.02] #todo: wieder rnd mu einfürgen
		mu[:,:,2] = [0.2 1; -0.01 0.0; 0.0 -0.03]
	else
		mu = zeros(m, n, M)
	end
	#beta = [1.1 0.9; 1.0 0.9; 0.9 0.8]
	beta = ones(m, n, M) + 0.1 * randn(m, n, M) #todo: wieder rnd beta einfügen
	beta[:,:,1] = [1.1 0.9; 1.0 0.9; 0.9 0.8]
	beta[:,:,2] = [1.2 0.9; 1.1 0.8; 0.9 0.7]
	rho = rho0 * ones(m, n, M)

	#initialize variables
	a = 0
	g = zeros(n,N)
	
	y = zeros(n,N,m,M)
	fp = zeros(m,N)
	Q = zeros(m,N)
	
	Lt = zeros(M,N)
	v = ones(M,N)
	z = ones(n,N,m,M)/N
	
	r = zeros(n,N,m,M)
	
	lambda = zeros(n,1)
	kappa = zeros(n,1)
	sigma2 = zeros(n,1)

	learnedParameters = GGParameters(alpha,beta,mu,rho)

	#originally initialized inside the loop
	LL = zeros(1,maxiter)
	ldet = zeros(M)
	dLL = zeros(1,maxiter)
	source_signals = zeros(n,N,M)

	for iter in 1:maxiter
        @show iter
		for h in 1:M
			ldet[h] =  -log(abs(det(A[:,:,h]))) #todo: in die get_ll funktion stecken
			Lt[h,:] .= log(proportions[h]) + ldet[h] #todo: same
			source_signals[:,:,:] = get_sources!(source_signals,A,x,h,M,n,centers)
			#Lt[iter,:] = get_likelihood_time(A, proportions, mu, beta, rho, alpha, b, h)
			#Lt[1,:] = [-84.2453 -40.6495 -9.3180 -7.9679 -38.9525 -83.2213]
			z, y = calculate_z_y!(m,n,learnedParameters,h,source_signals,y,Q,z)
			#Lt[h,:] = sum(loglikelihoodMMGG.(eachcol(mu[:,:,h]),eachcol(beta[:,:,h]),eachcol(rho[:,:,h]),eachrow(source_signals[:,:,h]),eachcol(alpha[:,:,h])))
			Lt[h,:] = calculate_Lt!(Lt[h,:], Q, y, n, m, h, learnedParameters)
			LL[iter] = calculate_LL(Lt, M, N, n)
		end
		@show Lt
		if iter > 1
			dLL[iter] = LL[iter] - LL[iter-1]
		end
		if iter > iterwin +1 #todo:testen
			lrate = calculate_lrate(dLL, lrate, lratefact, lnatrate, lratemax, mindll, iter,newt_start_iter, do_newton, iterwin)
			#lrate < 0 ? break : ""
			sdll = sum(dLL[iter-iterwin+1:iter])/iterwin
           # @show sdll
			if (sdll > 0) && (sdll < mindll)
				break
				
			end
		end
   

		vsum = zeros(M)
		for h in 1:M
			#update parameters
			if M > 1
				Lh = ones(M,N)
				for i in 1:M
					Lh[i,:] = Lt[h,:]
				end
				v[h,:] = 1 ./ sum(exp.(Lt-Lh),dims=1)
				vsum[h] = sum(v[h,:])
				proportions[h] = vsum[h] / N
			
				if proportions[h] == 0
					continue
				end
			end
			g, vsum, z, learnedParameters, kappa, lambda = update_parameters_and_other_stuff!(iter, v, proportions, vsum, h, M, N, m, n, Lt, fp, z, learnedParameters, lambda, rhomin, rhomax, update_rho, y, rholrate)
            
			if any(isnan, kappa) || any(isnan, source_signals) || any(isnan, lambda) || any(isnan, g) || any(isnan, alpha)
				println("NaN detected. Better stop.")
				@goto escape_from_NaN
			end
			#Newton
			A = newton_method(v, M, A, vsum, h, iter, source_signals, n, g, kappa, do_newton, newt_start_iter, lrate, lnatrate, N, sigma2, lambda)
		end
	
		A, learnedParameters, centers = reparameterize!(A, x, M, learnedParameters, v, centers, proportions, n)

		@show A
		#@show LL[iter]
        
	end

    @label escape_from_NaN

	for h in 1:M
		if M > 1
			centers[:,h] = centers[:,h] + mn #add mean back to model centers
		end
	end

	return z, A, Lt, LL
end


# #calculate z (which is u at first) lines 202 - 218
				#if M > 1 && m > 1
function calculate_z_y!(m,n,learnedParameters::GGParameters,h,source_signals,y,Q,z)
    for i in 1:n
        for j in 1:m
            y[i,:,j,h] = sqrt(learnedParameters.β[j,i,h]) * (source_signals[i,:,h] .- learnedParameters.μ[j,i,h])
            Q[j,:] .= log(learnedParameters.α[j,i,h]) + 0.5*log(learnedParameters.β[j,i,h]) .+ logpfun(y[i,:,j,h],learnedParameters.ρ[j,i,h])
        end
        if m > 1
            #hier ist eig. noch berechnung von Qmax und Lt
            for j in 1:m
                Qj = ones(m,1) .* Q[j,:]'
                z[i,:,j,h] = 1 ./ sum(exp.(Q-Qj),dims = 1)
            end
        end
    end
    return z, y
end


function newton_method(v, M, A, vsum, h, iter, source_signals, n, g, kappa, do_newton, newt_start_iter, lrate, lnatrate, N, sigma2, lambda)
	if M > 1
		sigma2 = source_signals[:,:,h].^2 * v[h,:] /vsum[h]
	else
		sigma2 = sum(source_signals.^2,dims=2) / N
	end
	dA = Matrix{Float64}(I, n, n) - g * source_signals[:,:,h]' 
	bflag = 0
	# if iter == 55
	# 	@show g
	# 	@show b
	# end
	#eig. in loop deklariert
	B = zeros(n,n)
	
	for i in 1:n
		for k = 1:n
			if i == k
				B[i,i] = dA[i,i] / (-0*dA[i,i] + lambda[i])#*0?? wtf??
			else
				denom = kappa[i]*kappa[k]*sigma2[i]*sigma2[k] - 1
				if denom > 0
					B[i,k] = (-kappa[k] * sigma2[i] * dA[i,k] + dA[k,i]) / denom
				else
					bflag = 1
				end
			end
		end		
	end
	if (bflag == 0) && (do_newton == 1) && (iter > newt_start_iter)
		A[:,:,h] = A[:,:,h] + lrate * A[:,:,h] * B
	else
		A[:,:,h] = A[:,:,h] - lnatrate * A[:,:,h] * dA
	end
	return A
end


function reparameterize!(A, x, M, learnedParameters::GGParameters, v, centers, proportions, n)
	mu = learnedParameters.μ
	beta = learnedParameters.β

	for h = 1:M
		if proportions[h] == 0
			continue
		end
		for i in 1:n
			tau = norm(A[:,i,h])
			A[:,i,h] = A[:,i,h] / tau
			mu[:,i,h] = mu[:,i,h] * tau
			beta[:,i,h] = beta[:,i,h] / tau^2
		end
	
		if M > 1
			cnew = x * v[h,:] /(sum(v[h,:]))
			for i in 1:n
				Wh = pinv(A[:,:,h])
				mu[:,i,h] = mu[:,i,h] .- Wh[i,:]' * (cnew-centers[:,h])
			end
			centers[:,h] = cnew
		end
	end

	learnedParameters.μ = mu
	learnedParameters.β = beta

	return A, learnedParameters, centers
end


function update_parameters_and_other_stuff!(iter, v, proportions, vsum, h, M, N, m, n, Lt, fp, z, learnedParameters::GGParameters, lambda, rhomin, rhomax, update_rho, y, rholrate)
	#it doesnt need iter, just there for test purposes
	alpha = learnedParameters.α
	beta = learnedParameters.β
	mu = learnedParameters.μ
	rho = learnedParameters.ρ

	g = zeros(n,N)
	kappa = zeros(n,1)
	eta = zeros(n,1)
	#sigma2 = zeros(n,1)
	
	
	#eigentlich in loop deklariert:
	zfp = zeros(m, N)
	
	for i in 1:n
		for j in 1:m
			sumz = 0
			if M > 1 #todo: testen
				if m > 1
					z[i,:,j,h] .= v[h,:] .* z[i,:,j,h]
					sumz = sum(z[i,:,j,h])
					alpha[j,i,h] = sumz / vsum[h]
				else
					z[i,:,j,h] = v[h,:]
					sumz = sum(z[i,:,j,h])
				end
			else
				if m > 1
					sumz = sum(z[i,:,j,h])
					alpha[j,i,h] = sumz / N
				else
					sumz = N
				end
			end
	
			if sumz > 0
				if (M > 1) | (m > 1)
					z[i,:,j,h] .= z[i,:,j,h] / sumz
				end
			else
				continue
			end
			#line 311
			fp[j,:] = ffun(y[i,:,j,h], rho[j,i,h])
			zfp[j,:] = z[i,:,j,h] .* fp[j,:]
			g[i,:] = g[i,:] .+ alpha[j,i,h] .* sqrt(beta[j,i,h]) .*zfp[j,:]
	
			kp = beta[j,i,h] .* sum(zfp[j,:].*fp[j,:])
	
			kappa[i] = kappa[i]  + alpha[j,i,h] * kp
	
			lambda[i] = lambda[i] + alpha[j,i,h] * (sum(z[i,:,j,h].*(fp[j,:].*y[i,:,j,h] .-1).^2) + mu[j,i,h]^2 * kp)
	
			if rho[j,i,h] <= 2
				if (m > 1) | (M > 1)
					dm = sum(zfp[j,:]./y[i,:,j,h])
					if dm > 0
						mu[j,i,h] = mu[j,i,h] + (1/sqrt(beta[j,i,h])) * sum(zfp[j,:]) / dm
					end
				end
	
				db = sum(zfp[j,:].*y[i,:,j,h])
				if db > 0
					beta[j,i,h] = beta[j,i,h] / db
				end
			else #todo: noch überprüfen, tritt bei erster iteration nicht ein
				if (m > 1) | (M > 1)
					if kp > 0
						mu[j,i,h] = mu[j,i,h] + sqrt(beta[j,i,h]) * sum(zfp[j,:]) / kp #only difference is sqrt instead of 1/sqrt
					end
				end
				db = (rho[j,i,h] * sum(z[i,:,j,h].*abs.(y[i,:,j,h]).^rho[j,i,h]))^(-2 / rho[j,i,h])
				beta[j,i,h] = beta[j,i,h] * db
			end
	
			if update_rho == 1
				ytmp = abs.(y[i,:,j,h]).^rho[j,i,h]
				dr = sum(z[i,:,j,h].*log.(ytmp).*ytmp)
	
				#todo: dieses if testen, wird bei ersten iteration nicht ausgeführt
				if rho[j,i,h] > 2
					dr2 = digamma(1+1/rho[j,i,h]) / rho[j,i,h] - dr
					if ~isnan(dr2) #todo: testen
						rho[j,i,h] = rho[j,i,h] + 0.5 * dr2
					end
				else
					dr2 = 1 - rho[j,i,h] * dr / digamma(1+1/rho[j,i,h])
					if ~isnan(dr2) #todo: testen
						rho[j,i,h] = rho[j,i,h] + rholrate *dr2
					end
				end
				rho[j,i,h] = min(rhomax, rho[j,i,h])
				rho[j,i,h] = max(rhomin, rho[j,i,h])
			end
		end
	end

	learnedParameters.α = alpha
	learnedParameters.β = beta
	learnedParameters.μ = mu
	learnedParameters.ρ = rho

	return g, vsum, z, learnedParameters, kappa, lambda
end