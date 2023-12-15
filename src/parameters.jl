#Normalizes source density location parameter (mu), scale parameter (beta) and model centers
function reparameterize!(myAmica::SingleModelAmica, data)
	(n,N) = size(myAmica.source_signals)
	mu = myAmica.learnedParameters.location
	beta = myAmica.learnedParameters.scale

	for i in 1:n
		tau = norm(myAmica.A[:,i])
		myAmica.A[:,i] ./= tau
		mu[:,i] .*= tau
		beta[:,i] ./= tau^2
	end

	myAmica.learnedParameters.location .= mu
	myAmica.learnedParameters.scale .= beta

	return myAmica
end

#Reparameterizes the parameters for the active models
function reparameterize!(myAmica::MultiModelAmica, data)
	(n,N) = size(myAmica.models[1].source_signals)
	M = size(myAmica.models,1)
	
	for h = 1:M
		mu = myAmica.models[h].learnedParameters.location
		beta = myAmica.models[h].learnedParameters.scale

		if myAmica.normalized_ica_weights[h] == 0
			continue
		end
		for i in 1:n
			tau = norm(myAmica.models[h].A[:,i])
			myAmica.models[h].A[:,i] = myAmica.models[h].A[:,i] / tau
			mu[:,i] = mu[:,i] * tau
			beta[:,i] = beta[:,i] / tau^2
		end
	
		if M > 1
			cnew = data * myAmica.ica_weights_per_sample[h,:] /(sum(myAmica.ica_weights_per_sample[h,:])) #todo: check why v not inverted
			for i in 1:n
				Wh = pinv(myAmica.models[h].A[:,:])
				mu[:,i] = mu[:,i] .- Wh[i,:]' * (cnew-myAmica.models[h].centers[:])
			end
			myAmica.models[h].centers = cnew
		end
		myAmica.models[h].learnedParameters.location .= mu
		myAmica.models[h].learnedParameters.scale .= beta
	end
end

#Calculates sum of z. Returns N if there is just one generalized Gaussian
@views function calculate_sumz(myAmica::SingleModelAmica,i,j)
	if myAmica.m == 1
		return size(myAmica.source_signals, 2)
	else
		return sum(myAmica.z[i,:,j])
	end
end

@views function calculate_sumz(myAmica::MultiModelAmica,i,j,h)
	return sum(myAmica.models[h].z[i,:,j])
end

#Calculates densities for each sample per ICA model and per Gaussian mixture
calculate_z!(myAmica::SingleModelAmica, i,j) = nothing
function calculate_z!(myAmica::MultiModelAmica, i,j,h)
	if myAmica.m > 1
		myAmica.models[h].z[i,:,j] .= myAmica.ica_weights_per_sample[h,:] .* myAmica.models[h].z[i,:,j]
	elseif myAmica.m == 1
		myAmica.models[h].z[i,:,j] .= myAmica.ica_weights_per_sample[h,:]
	end
end

#Updates the Gaussian mixture proportion parameter
function update_mixture_proportions!(sumz, myAmica::SingleModelAmica,j,i)
	N = size(myAmica.source_signals,2)
	if myAmica.m > 1
		myAmica.learnedParameters.proportions[j,i] = sumz / N
	end
end

function update_mixture_proportions!(sumz, myAmica::MultiModelAmica,j,i,h)
	if myAmica.m > 1
		myAmica.models[h].learnedParameters.proportions[j,i] = sumz / myAmica.ica_weights[h]
	end
end

#Updates the Gaussian mixture location parameter. Todo: merge again with MultiModel version
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

#Updates the Gaussian mixture scale parameter
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

#Sets the initial value for the shape parameter of the GeneralizedGaussians for each Model
function initialize_shape_parameter(myAmica::SingleModelAmica, shapelrate::LearningRate)
	myAmica.learnedParameters.shape .= shapelrate.init .*myAmica.learnedParameters.shape
end

function initialize_shape_parameter(myAmica::MultiModelAmica, shapelrate::LearningRate)
	M = size(myAmica.models,1)
	for h in 1:M
		myAmica.models[h].learnedParameters.shape .= shapelrate.init .*myAmica.models[h].learnedParameters.shape
	end
end

#Updates Gaussian mixture Parameters and mixing matrix. todo: rename since its not a loop for single model
function update_loop!(myAmica::SingleModelAmica, fp, lambda, shapelrate, update_shape, iter, do_newton, newt_start_iter, lrate)
		#Update parameters
		g, kappa = update_parameters!(myAmica, fp, lambda, shapelrate, update_shape)
		
		#Checks for NaN in parameters before updating the mixing matrix
		if any(isnan, kappa) || any(isnan, myAmica.source_signals) || any(isnan, lambda) || any(isnan, g) || any(isnan, myAmica.learnedParameters.proportions)
			throw(AmicaNaNException())
		end
		#Update mixing matrix via Newton method
		newton_method!(myAmica, iter, g, kappa, do_newton, newt_start_iter, lrate, lambda)
end

#Updates Gaussian mixture Parameters and mixing matrix.
function update_loop!(myAmica::MultiModelAmica, fp, lambda, shapelrate, update_shape, iter, do_newton, newt_start_iter, lrate)
	(n,N) = size(myAmica.models[1].source_signals)
	M = size(myAmica.models,1)

	myAmica.ica_weights_per_sample = ones(M,N)
	for h in 1:M
		#Calcutes ICA model weights
		myAmica.ica_weights_per_sample[h,:] = zeros(N)
		for i in 1:M
			myAmica.ica_weights_per_sample[h,:] = myAmica.ica_weights_per_sample[h,:] + exp.(myAmica.models[i].Lt-myAmica.models[h].Lt)
		end
		myAmica.ica_weights_per_sample[h,:] = 1 ./ myAmica.ica_weights_per_sample[h,:]
		myAmica.ica_weights[h] = sum(myAmica.ica_weights_per_sample[h,:])
		myAmica.normalized_ica_weights[h] = myAmica.ica_weights[h] / N
		
		#If model weight equals 0 skip update for this model
		if myAmica.normalized_ica_weights[h] == 0
			continue
		end

		g, kappa, lambda = update_parameters!(myAmica, h, fp, lambda, shapelrate, update_shape)#todo: remove return

		#Checks for NaN in parameters before updating the mixing matrix
		if any(isnan, kappa) || any(isnan, myAmica.models[h].source_signals) || any(isnan, lambda) || any(isnan, g) || any(isnan, myAmica.models[h].learnedParameters.proportions)
			throw(AmicaNaNException())
		end
		#Update mixing matrix via Newton method
		newton_method!(myAmica, h, iter, g, kappa, do_newton, newt_start_iter, lrate, lambda)
	end
end

#Updates Gaussian mixture parameters. It also returns g, kappa and lamda which are needed to apply the newton method.
#Todo: Save g, kappa, lambda in structure, remove return
@views function update_parameters!(myAmica::SingleModelAmica, fp, lambda, lrate_rho::LearningRate, update_shape)
	alpha = myAmica.learnedParameters.proportions
	beta = myAmica.learnedParameters.scale
	mu = myAmica.learnedParameters.location
	rho = myAmica.learnedParameters.shape
	(n,N) = size(myAmica.source_signals)
	m = myAmica.m
	g = zeros(n,N)
	kappa = zeros(n,1)
	zfp = zeros(m, N)

	# update myAmica.learnedParameters.proportions & myAmica.z
	# depends on 
	# - myAmica.z
	# - myAmica.source_signals
	for i in 1:n
		for j in 1:m
			sumz = calculate_sumz(myAmica,i,j)
			update_mixture_proportions!(sumz,myAmica,j,i)
			if sumz > 0
				if (m > 1)
					myAmica.z[i,:,j] .= myAmica.z[i,:,j] / sumz
				end
			else
				continue
			end
		end
	end
	
	# update 
	# - fp
	# - zfp
	# - g
	# - kp
	# - kappa
	# - lambda
	# - mu

	# depends on 
	# - myAmica.y
	# - myAmica.z
	# - rho
	# - alpha
	# - beta
	# - mu
	for i in 1:n
		for j in 1:m
			fp[j,:] .= ffun(myAmica.y[i,:,j], rho[j,i])

			zfp[j,:] .= myAmica.z[i,:,j] .* fp[j,:]
			g[i,:] .+= alpha[j,i] .* sqrt(beta[j,i]) .*zfp[j,:]
	
			kp = beta[j,i] .* sum(zfp[j,:].*fp[j,:])
	
			kappa[i] += alpha[j,i] * kp
	
			lambda[i] += alpha[j,i] .* (sum(myAmica.z[i,:,j].*(fp[j,:].*myAmica.y[i,:,j] .-1).^2) .+ mu[j,i]^2 .* kp)
			mu[j,i] = update_location(myAmica,rho[j,i],zfp[j,:],myAmica.y[i,:,j],mu[j,i],beta[j,i],kp)


			beta[j,i] = update_scale(zfp[j,:],myAmica.y[i,:,j],beta[j,i],myAmica.z[i,:,j],rho[j,i])

		end
	end

	# update rho
	# depends on rho, zfp, myAmica.y, mu, beta, kp
	for i in 1:n
		for j in 1:m
			if update_shape == 1
				update_shape!(myAmica, rho, j, i, lrate_rho)
			end
		end
	end


	myAmica.learnedParameters.proportions = alpha
	myAmica.learnedParameters.scale = beta
	myAmica.learnedParameters.location = mu
	myAmica.learnedParameters.shape = rho

	return g, kappa
end

#Updates Gaussian mixture parameters. It also returns g, kappa and lamda which are needed to apply the newton method.
#Todo: Save g, kappa, lambda in structure, remove return
@views function update_parameters!(myAmica::MultiModelAmica, h, fp, lambda, lrate_rho::LearningRate, update_shape)
	alpha = myAmica.models[h].learnedParameters.proportions #todo: move into loop and add h
	beta = myAmica.models[h].learnedParameters.scale
	mu = myAmica.models[h].learnedParameters.location
	rho = myAmica.models[h].learnedParameters.shape

	(n,N) = size(myAmica.models[1].source_signals)
	m = myAmica.m
	g = zeros(n,N)
	kappa = zeros(n,1)
	zfp = zeros(m, N)

	
	#=Threads.@threads=# for i in 1:n
		for j in 1:m
			sumz = 0
			calculate_z!(myAmica, i, j, h)
			sumz = calculate_sumz(myAmica,i,j,h)
			update_mixture_proportions!(sumz,myAmica,j,i,h)
			if sumz > 0
				myAmica.models[h].z[i,:,j] .= myAmica.models[h].z[i,:,j] / sumz
			else
				continue
			end
	
			fp[j,:] .= ffun(myAmica.models[h].y[i,:,j], rho[j,i])
			zfp[j,:] .= myAmica.models[h].z[i,:,j] .* fp[j,:]
			g[i,:] .+= alpha[j,i] .* sqrt(beta[j,i]) .*zfp[j,:]
	
			kp = beta[j,i] .* sum(zfp[j,:].*fp[j,:])
	
			kappa[i] += alpha[j,i] * kp
	
			lambda[i] += alpha[j,i] .* (sum(myAmica.models[h].z[i,:,j].*(fp[j,:].*myAmica.models[h].y[i,:,j] .-1).^2) .+ mu[j,i]^2 .* kp)
			mu[j,i] = update_location(myAmica,rho[j,i],zfp[j,:],myAmica.models[h].y[i,:,j],mu[j,i],beta[j,i],kp)
			
			
			beta[j,i] = update_scale(zfp[j,:],myAmica.models[h].y[i,:,j],beta[j,i],myAmica.models[h].z[i,:,j],rho[j,i])

			if update_shape == 1
				update_shape!(myAmica.models[h], rho, j, i, lrate_rho) #uses SingleModel version on each model
			end
		end
	end
	myAmica.models[h].learnedParameters.proportions = alpha
	myAmica.models[h].learnedParameters.scale = beta
	myAmica.models[h].learnedParameters.location = mu
	myAmica.models[h].learnedParameters.shape = rho
	return g, kappa
end

#Updates the Gaussian mixture shape parameter
@views function update_shape!(myAmica::SingleModelAmica, rho, j, i, lrate_rho::LearningRate)
	rhomin, rhomax, shapelrate = lrate_rho.minimum, lrate_rho.maximum, lrate_rho.lrate
	ytmp = optimized_pow(abs.(myAmica.y[i,:,j]), rho[j,i])
	dr = sum(myAmica.z[i,:,j].*optimized_log(ytmp).*ytmp)

	if rho[j,i] > 2
		dr2 = digamma(1+1/rho[j,i]) / rho[j,i] - dr
		if ~isnan(dr2)
			rho[j,i] += 0.5 * dr2
		end
	else
		dr2 = 1 - rho[j,i] * dr / digamma(1+1/rho[j,i])
		if ~isnan(dr2)
			rho[j,i] += shapelrate *dr2
		end
	end

	rho[j,i] = clamp(rho[j,i], rhomin, rhomax)
end

#Updates the Gaussian mixture shape parameter. MultiModel version no longer in use
function update_shape!(myAmica::MultiModelAmica, rho, j, i, h, lrate_rho::LearningRate)
	rhomin, rhomax, shapelrate = lrate_rho.minimum, lrate_rho.maximum, lrate_rho.lrate
	ytmp = abs.(myAmica.models[h].y[i,:,j]).^rho[j,i]
	dr = sum(myAmica.models[h].z[i,:,j].*log.(ytmp).*ytmp)

	if rho[j,i] > 2
		dr2 = digamma(1+1/rho[j,i]) / rho[j,i] - dr
		if ~isnan(dr2)
			rho[j,i] = rho[j,i] + 0.5 * dr2
		end
	else
		dr2 = 1 - rho[j,i] * dr / digamma(1+1/rho[j,i])
		if ~isnan(dr2)
			rho[j,i] = rho[j,i] + shapelrate *dr2
		end
	end
	rho[j,i] = min(rhomax, rho[j,i])
	rho[j,i] = max(rhomin, rho[j,i])
end

#Calculates determinant of mixing Matrix A (with log). first log-likelihood part of L = |A| * p(sources)
function calculate_ldet!(myAmica::SingleModelAmica)
	#myAmica.ldet = -log(abs(det(myAmica.A)))
	myAmica.ldet = -logabsdet(myAmica.A)[1]
	@debug :ldet myAmica.ldet
end

function calculate_ldet!(myAmica::MultiModelAmica)
	for h in 1:length(myAmica.models)
		myAmica.models[h].ldet = -log(abs(det(myAmica.models[h].A)))
	end
end

#Updates source singal estimations by unmixing the data
function update_sources!(myAmica::SingleModelAmica, data)
	b = pinv(myAmica.A) * data
	myAmica.source_signals = b
end

function update_sources!(myAmica::MultiModelAmica, data)
	n = size(myAmica.models[1].A, 1)
	for h in 1:length(myAmica.models)
		for i in 1:n 
			Wh = pinv(myAmica.models[h].A)
			myAmica.models[h].source_signals[i,:] = Wh[i,:]' * data .- Wh[i,:]' * myAmica.models[h].centers
		end
	end
end

#Adjusts learning rate depending on log-likelihood growth during past iterations. How many depends on iterwin. Uses LearningRate type from types.jl
function calculate_lrate!(dLL, lrateType::LearningRate, iter, newt_start_iter, do_newton, iterwin)

	lratefact,lnatrate,lratemax, = lrateType.decreaseFactor, lrateType.natural_rate, lrateType.maximum
	lrate = lrateType.lrate
	sdll = sum(dLL[iter-iterwin+1:iter])/iterwin

    if sdll < 0
        println("Likelihood decreasing!")
        lrate = lrate * lratefact
    else
        if (iter > newt_start_iter) && do_newton == 1
            lrate = min(lratemax,lrate + min(0.1,lrate))
        else
            lrate = min(lnatrate,lrate + min(0.1,lrate))
        end
    end
	lrateType.lrate = lrate
end