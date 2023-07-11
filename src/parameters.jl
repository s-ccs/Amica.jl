function reparameterize!(myAmica, data, v)
	n = myAmica.n
	M = myAmica.M
	mu = myAmica.learnedParameters.location
	beta = myAmica.learnedParameters.scale

	for h = 1:M
		if myAmica.proportions[h] == 0
			continue
		end
		for i in 1:n
			tau = norm(myAmica.A[:,i,h])
			myAmica.A[:,i,h] = myAmica.A[:,i,h] / tau
			mu[:,i,h] = mu[:,i,h] * tau
			beta[:,i,h] = beta[:,i,h] / tau^2
		end
	
		if M > 1
			cnew = data * v[h,:] /(sum(v[h,:]))
			for i in 1:n
				Wh = pinv(myAmica.A[:,:,h])
				mu[:,i,h] = mu[:,i,h] .- Wh[i,:]' * (cnew-myAmica.centers[:,h])
			end
			myAmica.centers[:,h] = cnew
		end
	end

	myAmica.learnedParameters.location .= mu
	myAmica.learnedParameters.scale .= beta

	return myAmica
end

#todo: all of those
function update_parameters!(myAmica, v, vsum, h, fp, lambda, lrate_rho::LearningRate, update_rho)
	update_prop()
	update_scale()
	update_location()
	update_shape()
	myAmica.z[i,:,j,h] .= calculate_z(myAmica, z[i,:,j,h])
	calculate_g()
end


#todo: need different versions for Multi & Single Models
function update_parameters_and_other_stuff!(myAmica, v, vsum, h, fp, lambda, lrate_rho::LearningRate, update_rho)
	M = myAmica.M
	N = myAmica.N
	n = myAmica.n 
	m = myAmica.m
	alpha = myAmica.learnedParameters.prop
	beta = myAmica.learnedParameters.scale
	mu = myAmica.learnedParameters.location
	rho = myAmica.learnedParameters.shape

	g = zeros(n,N)
	kappa = zeros(n,1)
	eta = zeros(n,1)
	#sigma2 = zeros(n,1)
	
	
	#eigentlich in loop deklariert:
	zfp = zeros(m, N)
	
	for i in 1:n
		for j in 1:m
			sumz = 0
			if M > 1
				if m > 1
					myAmica.z[i,:,j,h] .= v[h,:] .* myAmica.z[i,:,j,h]
					sumz = sum(myAmica.z[i,:,j,h])
					alpha[j,i,h] = sumz / vsum[h]
				else
					myAmica.z[i,:,j,h] = v[h,:]
					sumz = sum(myAmica.z[i,:,j,h])
				end
			else
				if m > 1
					sumz = sum(myAmica.z[i,:,j,h])
					alpha[j,i,h] = sumz / N
				else
					sumz = N
				end
			end
	
			if sumz > 0
				if (M > 1) | (m > 1)
					myAmica.z[i,:,j,h] .= myAmica.z[i,:,j,h] / sumz
				end
			else
				continue
			end
			#line 311
			fp[j,:] = ffun(myAmica.y[i,:,j,h], rho[j,i,h])
			zfp[j,:] = myAmica.z[i,:,j,h] .* fp[j,:]
			g[i,:] = g[i,:] .+ alpha[j,i,h] .* sqrt(beta[j,i,h]) .*zfp[j,:]
	
			kp = beta[j,i,h] .* sum(zfp[j,:].*fp[j,:])
	
			kappa[i] = kappa[i]  + alpha[j,i,h] * kp
	
			lambda[i] = lambda[i] + alpha[j,i,h] * (sum(myAmica.z[i,:,j,h].*(fp[j,:].*myAmica.y[i,:,j,h] .-1).^2) + mu[j,i,h]^2 * kp)
	
			if rho[j,i,h] <= 2
				if (m > 1) | (M > 1)
					dm = sum(zfp[j,:]./myAmica.y[i,:,j,h])
					if dm > 0
						mu[j,i,h] = mu[j,i,h] + (1/sqrt(beta[j,i,h])) * sum(zfp[j,:]) / dm
					end
				end
	
				db = sum(zfp[j,:].*myAmica.y[i,:,j,h])
				if db > 0
					beta[j,i,h] = beta[j,i,h] / db
				end
			else
				if (m > 1) | (M > 1)
					if kp > 0
						mu[j,i,h] = mu[j,i,h] + sqrt(beta[j,i,h]) * sum(zfp[j,:]) / kp #only difference is sqrt instead of 1/sqrt
					end
				end
				db = (rho[j,i,h] * sum(myAmica.z[i,:,j,h].*abs.(myAmica.y[i,:,j,h]).^rho[j,i,h]))^(-2 / rho[j,i,h])
				beta[j,i,h] = beta[j,i,h] * db
			end
	
			if update_rho == 1
				update_rho!(myAmica, rho, j, i, h, lrate_rho)
			end
		end
	end

	myAmica.learnedParameters.prop = alpha
	myAmica.learnedParameters.scale = beta
	myAmica.learnedParameters.location = mu
	myAmica.learnedParameters.shape = rho

	return myAmica, g, vsum, kappa, lambda
end

function update_rho!(myAmica::MultiModelAmica, rho, j, i, h, lrate_rho::LearningRate)
	rhomin, rhomax, rholrate = lrate_rho.minimum, lrate_rho.maximum, lrate_rho.lrate
	ytmp = abs.(myAmica.y[i,:,j,h]).^rho[j,i,h]
	dr = sum(myAmica.z[i,:,j,h].*log.(ytmp).*ytmp)

	if rho[j,i,h] > 2
		dr2 = digamma(1+1/rho[j,i,h]) / rho[j,i,h] - dr
		if ~isnan(dr2)
			rho[j,i,h] = rho[j,i,h] + 0.5 * dr2
		end
	else
		dr2 = 1 - rho[j,i,h] * dr / digamma(1+1/rho[j,i,h])
		if ~isnan(dr2)
			rho[j,i,h] = rho[j,i,h] + rholrate *dr2
		end
	end
	rho[j,i,h] = min(rhomax, rho[j,i,h])
	rho[j,i,h] = max(rhomin, rho[j,i,h])
end