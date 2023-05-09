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
	c = zeros(n,M)
	eye = Matrix{Float64}(I, n, n)

	for h in 1:M #todo: wieder randomisieren
		# A[:,:,h] = eye[n] .+ 0.1*rand(n,n)
		# for i in 1:n
		# 	A[:,i,h] = A[:,i,h] / norm(A[:,i,h])
		# end
		A = [1.0 0.003; -0.05 1.0]
		c[:,h] = zeros(n,1)
	end

	gm = (1/M) * ones(M,1)
	alpha = (1/m) * ones(m,n,M)
	
	if m > 1
		#mu = 0.1 * randn(m, n, M)
		mu = [0.1 0.9; -0.01 0.0; 0.0 -0.02] #todo: wieder rnd mu einfürgen
	else
		mu = zeros(m, n, M)
	end

	#beta = ones(m, n, M) + 0.1 * randn(m, n, M) todo: wieder rnd beta einfügen
	beta = [1.1 0.9; 1.0 0.9; 0.9 0.8]
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

	#originally initialized inside the loop
	LL = zeros(1,maxiter)
	ldet = zeros(M)
	dLL = zeros(1,maxiter)
	b = zeros(n,N,M)

	counter = 0
	for iter in 1:maxiter
        @show iter
		for h in 1:M
			ldet[h] =  -log(abs(det(A[:,:,h]))) #todo: in die get_ll funktion stecken
			Lt[h,:] .= log(gm[h]) + ldet[h] #todo: same
			b .= get_sources!(b,A,x,h,M,n)
			#Lt[iter,:] = get_likelihood_time(A, gm, mu, beta, rho, alpha, b, h)
			#Lt[1,:] = [-84.2453 -40.6495 -9.3180 -7.9679 -38.9525 -83.2213]
			Lt[h,:] = sum(loglikelihoodMMGG.(eachcol(mu[:,:,h]),eachcol(beta[:,:,h]),eachcol(rho[:,:,h]),eachrow(b[:,:,h]),eachcol(alpha[:,:,h])))
			LL[iter] = calculate_LL(Lt, M, N, n)
	
			z, y = calculate_z_y(m,n,beta,mu,alpha,rho,h,b,y,Q,z)
		end
		if iter > 1
			dLL[iter] = LL[iter] - LL[iter-1]
		end
		if iter > iterwin +1 #todo:testen
			lrate = calculate_lrate(dLL, lrate, lratefact, lnatrate,lratemax,mindll,iter,newt_start_iter,do_newton, iterwin)
			#lrate < 0 ? break : ""
			sdll = sum(dLL[iter-iterwin+1:iter])/iterwin
            @show sdll
			if (sdll > 0) && (sdll < mindll)
				counter = 1
				break
				
			end
		end
   
		
		vsum = zeros(M)
		for h in 1:M
			#update parameters
            
			g, vsum, z, alpha, beta, kappa, lambda, mu, rho = update_parameters_and_other_stuff(iter, vsum, h, M, N, m, n, Lt, fp, z, alpha, beta, kappa, lambda, mu, rho, rhomin, rhomax, update_rho, y, g, rholrate)
            
			#Newton
            
			A = newton_method(M, A, vsum, h, iter, b, n, g, kappa, do_newton, newt_start_iter, lrate, lnatrate, N, sigma2, lambda)
            
		end
	
		A, mu, beta, c = reparameterize(A, x, M, mu, beta, v, c, gm, n)
        
	end
    
	for h in 1:M
		if M > 1
			c[:,h] = c[:,h] + mn #add mean back to model centers
		end
	end
	return z, counter, A, Lt, LL
end


# #calculate z (which is u at first) lines 202 - 218, probably unnessary
				#if M > 1 && m > 1
function calculate_z_y(m,n,beta,mu,alpha,rho,h,b,y,Q,z)
    for i in 1:n
        for j in 1:m
            y[i,:,j,h] = sqrt(beta[j,i,h]) * (b[i,:,h] .- mu[j,i,h])
            Q[j,:] .= log(alpha[j,i,h]) + 0.5*log(beta[j,i,h]) .+ logpfun(y[i,:,j,h],rho[j,i,h])
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


function newton_method(M, A, vsum, h, iter, b, n, g, kappa, do_newton, newt_start_iter, lrate, lnatrate, N, sigma2, lambda)
	if M > 1 #todo:testen
		sigma2 = b[:,:,h].^2 * v[h,:]'/vsum(h)
	else
		sigma2 = sum(b.^2,dims=2) / N
	end
	dA = Matrix{Float64}(I, n, n) - g * b[:,:,h]' 
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

# ╔═╡ b3637deb-985f-4fea-bf99-d956ad4a2877

#function reparameterize(A::AbstractArray,...)
#    for h in 1:size(A,4)
#        A[:,:,:,h],... = reparameterize(A[:,:,:,h],...)
#    end
#end
function reparameterize(A, x, M, mu, beta, v, c, gm, n)
	for h = 1:M
		if gm[h] == 0 #todo:  das auch wieder einfügen
			#continue
		end
		for i in 1:n
			tau = norm(A[:,i,h])
			A[:,i,h] = A[:,i,h] / tau
			mu[:,i,h] = mu[:,i,h] * tau
			beta[:,i,h] = beta[:,i,h] / tau^2
		end
	
		if M > 1 #todo: testen
			cnew = x * v[h,:]'/(sum(v[h,:]))
			for i in 1:n
				Wh = pinv(A[:,:,h])
				mu[:,i,h] = mu[:,i,h] - Wh[i,:]*(cnew-c[:,h])
			end
			c[:,h] = cnew
		end
	end
	return A, mu, beta, c
end


function update_parameters_and_other_stuff(iter, vsum, h, M, N, m, n, Lt, fp, z, alpha, beta, kappa, lambda, mu, rho, rhomin, rhomax, update_rho, y, g, rholrate)
	#it doesnt need iter, just there for test purposes
	if M > 1 #todo: testen
		Lh = ones(M,N)
		for i in 1:M
			Lh[i,:] = Lt[h,:]
		end
		v[h,:] = 1 ./ sum(exp.(Lt-Lh),dims=1)
		vsum[h] = sum(v[h,:])
		gm[h] = vsum[h] / N
	
		if gm[h] == 0
			#continue
		end
	end
	#g = zeros(n,N)
	#kappa = zeros(n,1)
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
			g[i,:] = g[i,:] .+alpha[j,i,h] .*sqrt(beta[j,i,h]) .*zfp[j,:]
	
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
	return g, vsum, z, alpha, beta, kappa, lambda, mu, rho
end