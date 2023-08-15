#normalizes source density location parameter (mu), scale parameter (beta) and model centers
#todo: remove third index for singlemodel on the arrays
function reparameterize!(myAmica::SingleModelAmica, data, v)
	n = myAmica.n
	#M = myAmica.M
	M = 1#todo: remove
	mu = myAmica.learnedParameters.location
	beta = myAmica.learnedParameters.scale

	for i in 1:n
		tau = norm(myAmica.A[:,i])
		myAmica.A[:,i] = myAmica.A[:,i] / tau
		mu[:,i] = mu[:,i] * tau
		beta[:,i] = beta[:,i] / tau^2
	end

	if M > 1
		cnew = data * v[:] /(sum(v[:]))
		for i in 1:n
			Wh = pinv(myAmica.A[:,:])
			mu[:,i] = mu[:,i] .- Wh[i,:]' * (cnew-myAmica.centers[:])
		end
		myAmica.centers[:] = cnew
	end

	myAmica.learnedParameters.location .= mu
	myAmica.learnedParameters.scale .= beta

	return myAmica
end

function reparameterize!(myAmica::MultiModelAmica, data, v)
	n = myAmica.n
	M = myAmica.M
	
	for h = 1:M
		mu = myAmica.models[h].learnedParameters.location
		beta = myAmica.models[h].learnedParameters.scale

		if myAmica.models[h].proportions == 0
			continue
		end
		for i in 1:n
			tau = norm(myAmica.models[h].A[:,i])
			myAmica.models[h].A[:,i] = myAmica.models[h].A[:,i] / tau
			mu[:,i] = mu[:,i] * tau
			beta[:,i] = beta[:,i] / tau^2
		end
	
		if M > 1
			cnew = data * v[h,:] /(sum(v[h,:]))
			for i in 1:n
				Wh = pinv(myAmica.models[h].A[:,:])
				mu[:,i] = mu[:,i] .- Wh[i,:]' * (cnew-myAmica.models[h].centers[:])
			end
			myAmica.centers[:,h] = cnew
		end
		myAmica.models[h].learnedParameters.location .= mu
		myAmica.models[h].learnedParameters.scale .= beta
	end


	return myAmica
end

function calculate_sumz(z, myAmica::SingleModelAmica)
	if myAmica.m == 1
		return myAmica.N
	else
		return sum(z)
	end
end

function calculate_sumz(z, myAmica::MultiModelAmica)
		return sum(z)
end


calculate_z!(myAmica::SingleModelAmica, v, z, i,j) = nothing
function calculate_z!(myAmica::MultiModelAmica, v, z, i,j,h)
	if myAmica.m > 1
		myAmica.models[h].z[i,:,j] .= v .* z
	elseif myAmica.m == 1
		myAmica.models.z[i,:,j] = v
	end
end

function update_mixture_proportions!(proportions, sumz, vsum, myAmica::SingleModelAmica,j,i)
	if myAmica.m > 1
		myAmica.learnedParameters.prop[j,i] = sumz / myAmica.N
	end
end

function update_mixture_proportions!(proportions, sumz, vsum, myAmica::MultiModelAmica,j,i,h)
	if myAmica.m > 1
		myAmica.models[h].learnedParameters.prop[j,i] = sumz / vsum #todo: rename in mixture_prop
	end
end

function update_location(myAmica::SingleModelAmica,shape,zfp,y,location,scale,kp)
	m = myAmica.m
	if shape <= 2
		if (m > 1) 
			dm = sum(zfp./y)
			if dm > 0
				return location + (1/sqrt(scale)) * sum(zfp) / dm
			end
		end
	else
		if (m > 1) && kp > 0
			return location + sqrt(scale) * sum(zfp) / kp
		end
	end
	return location
end
#todo: wieder zusammen nehmen
function update_location(myAmica::MultiModelAmica,shape,zfp,y,location,scale,kp)
	if shape <= 2
			dm = sum(zfp./y)
			if dm > 0
				return location + (1/sqrt(scale)) * sum(zfp) / dm
			end
	else
		if kp > 0
			return location + sqrt(scale) * sum(zfp) / kp
		end
	end
	return location
end

function update_scale(zfp,y,scale,z,shape)
	if shape <= 2
		db = sum(zfp.*y)
		if db > 0
			scale = scale ./ db
		end
	else
		db = (shape .* sum(z.*abs.(y).^shape)).^(.- 2 ./ shape)
		scale = scale .* db
	end
	return scale
end

function update_parameters!(myAmica::SingleModelAmica, v, vsum, fp, lambda, lrate_rho::LearningRate, update_rho)
	alpha = myAmica.learnedParameters.prop
	beta = myAmica.learnedParameters.scale
	mu = myAmica.learnedParameters.location
	rho = myAmica.learnedParameters.shape
	#M = myAmica.M
	M = 1 #todo: remove
	N = myAmica.N
	n = myAmica.n 
	m = myAmica.m
	g = zeros(n,N)
	kappa = zeros(n,1)
	zfp = zeros(m, N)

	
	Threads.@threads for i in 1:myAmica.n
		for j in 1:myAmica.m
			sumz = 0
			calculate_z!(myAmica, v[:], myAmica.z[i,:,j], i, j) #todo: check if necessary in singleModel
			sumz = calculate_sumz(myAmica.z[i,:,j], myAmica)
			update_mixture_proportions!(myAmica.learnedParameters.prop[j,i],sumz, vsum,myAmica,j,i)
			if sumz > 0
				if (M > 1) | (m > 1)
						myAmica.z[i,:,j] .= myAmica.z[i,:,j] / sumz
				end
			else
				continue
			end
	
			fp[j,:] = ffun(myAmica.y[i,:,j], rho[j,i])
			zfp[j,:] = myAmica.z[i,:,j] .* fp[j,:]
			g[i,:] = g[i,:] .+ alpha[j,i] .* sqrt(beta[j,i]) .*zfp[j,:]
	
			kp = beta[j,i] .* sum(zfp[j,:].*fp[j,:])
	
			kappa[i] = kappa[i]  + alpha[j,i] * kp
	
			lambda[i] = lambda[i] .+ alpha[j,i] .* (sum(myAmica.z[i,:,j].*(fp[j,:].*myAmica.y[i,:,j] .-1).^2) .+ mu[j,i]^2 .* kp)
			mu[j,i] =  update_location(myAmica,rho[j,i],zfp[j,:],myAmica.y[i,:,j],mu[j,i],beta[j,i],kp)
			
			
			beta[j,i] = update_scale(zfp[j,:],myAmica.y[i,:,j],beta[j,i],myAmica.z[i,:,j],rho[j,i])

			if update_rho == 1
				update_rho!(myAmica, rho, j, i, lrate_rho)
			end
		end
	end
	myAmica.learnedParameters.prop = alpha
	myAmica.learnedParameters.scale = beta
	myAmica.learnedParameters.location = mu
	myAmica.learnedParameters.shape = rho
	return myAmica, g, kappa, lambda
end

function update_parameters!(myAmica::MultiModelAmica, v, vsum, h, fp, lambda, lrate_rho::LearningRate, update_rho)
	alpha = myAmica.learnedParameters.prop #todo: move into loop and add h
	beta = myAmica.learnedParameters.scale
	mu = myAmica.learnedParameters.location
	rho = myAmica.learnedParameters.shape
	M = myAmica.M
	N = myAmica.N
	n = myAmica.n 
	m = myAmica.m
	g = zeros(n,N)
	kappa = zeros(n,1)
	zfp = zeros(m, N)

	
	Threads.@threads for i in 1:myAmica.n
		for j in 1:myAmica.m
			sumz = 0
			calculate_z!(myAmica, v[h,:], myAmica.models[h].z[i,:,j], i, j, h)
			sumz = calculate_sumz(myAmica.models[h].z[i,:,j], myAmica)
			update_mixture_proportions!(myAmica.models[h].learnedParameters.prop[j,i],sumz, vsum[h],myAmica,j,i,h)
			if sumz > 0
				if (M > 1) | (m > 1)
						myAmica.models[h].z[i,:,j] .= myAmica.models[h].z[i,:,j] / sumz
				end
			else
				continue
			end
	
	
			fp[j,:] = ffun(myAmica.models[h].y[i,:,j], rho[j,i])
			zfp[j,:] = myAmica.models[h].z[i,:,j] .* fp[j,:]
			g[i,:] = g[i,:] .+ alpha[j,i,h] .* sqrt(beta[j,i,h]) .*zfp[j,:]
	
			kp = beta[j,i,h] .* sum(zfp[j,:].*fp[j,:])
	
			kappa[i] = kappa[i]  + alpha[j,i,h] * kp
	
			lambda[i] = lambda[i] .+ alpha[j,i,h] .* (sum(myAmica.z[i,:,j,h].*(fp[j,:].*myAmica.y[i,:,j,h] .-1).^2) .+ mu[j,i,h]^2 .* kp)
			mu[j,i,h] =  update_location(myAmica,rho[j,i,h],zfp[j,:],myAmica.models[h].y[i,:,j],mu[j,i,h],beta[j,i,h],kp)
			
			
			beta[j,i,h] = update_scale(zfp[j,:],myAmica.models[h].y[i,:,j],beta[j,i,h],myAmica.models[h].z[i,:,j],rho[j,i,h])

			if update_rho == 1
				update_rho!(myAmica, rho, j, i, h, lrate_rho)
			end
		end
	end
	myAmica.learnedParameters.prop = alpha
	myAmica.learnedParameters.scale = beta
	myAmica.learnedParameters.location = mu
	myAmica.learnedParameters.shape = rho
	return myAmica, g, kappa, lambda
end

function update_rho!(myAmica::SingleModelAmica, rho, j, i, lrate_rho::LearningRate)
	h = 1 #todo: remove
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
