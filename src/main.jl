"""
Main AMICA algorithm

"""
function fit!(amica::AbstractAmica,x;kwargs...)
	amica!(amica,x;kwargs...)
end

function fit(amicaType::Type{T},x;M=1,remove_mean = true,kwargs...) where {T<:AbstractAmica}
	if remove_mean
		removeMean(x) # TODO: removeMean! as it is inplace
	end
	amica = T(x;M=M)
	fit!(amica,x;kwargs...)
	return amica
end

function amica!(myAmica::AbstractAmica,
	x;
	lrate = LearningRate(),
	rholrate = LearningRate(;lrate = 0.1,minimum=0.5,maximum=5,init=1.5),
	dispinterval = 50,
	showLL = 1,
	plotfig = 1,
	plothist = 1,
	nbins = 50,
	show_progress = true,
	maxiter = 500,
	do_newton = 1,
	newt_start_iter = 1,# TODO Check
	iterwin = 10,
	update_rho = 1,
	mindll = 1e-8,

	kwargs...)
	
	myAmica.learnedParameters.ρ .= rholrate.init .*myAmica.learnedParameters.ρ
	#learnedParameters(m::AbstractAmica) = m.learnedParameters
	

	M = myAmica.M
	n = myAmica.n
	N = myAmica.N
	m = myAmica.m


	#Mx = maximum(abs.(x)) #maximum and max are not the same

	mn = mean(x, dims = 2) #should be zeros if remove_mean = 0
	#a = 0
	g = zeros(n,N)
	v = ones(M,N)
	lambda = zeros(n,1)
	kappa = zeros(n,1)
	sigma2 = zeros(n,1)

	dLL = zeros(1,maxiter)

	fp = zeros(m,N)

	#r = zeros(n,N,m,M)
	
	lambda = zeros(n,1)
	kappa = zeros(n,1)
	sigma2 = zeros(n,1)

    prog = ProgressUnknown("Minimizing"; showspeed=true)

	for iter in 1:maxiter
        
		for h in 1:M
			myAmica = get_sources!(myAmica, x, h)
			#Lt[iter,:] = get_likelihood_time(A, proportions, mu, beta, rho, alpha, b, h)
			#Lt[1,:] = [-84.2453 -40.6495 -9.3180 -7.9679 -38.9525 -83.2213]
			myAmica = calculate_z_y!(myAmica,h)
			#Lt[h,:] = sum(loglikelihoodMMGG.(eachcol(mu[:,:,h]),eachcol(beta[:,:,h]),eachcol(rho[:,:,h]),eachrow(source_signals[:,:,h]),eachcol(alpha[:,:,h])))
			myAmica = calculate_Lt!(myAmica, h)
			myAmica = calculate_LL!(myAmica, iter)
		end
		
		if iter > 1
			dLL[iter] = myAmica.LL[iter] - myAmica.LL[iter-1]
		end
		if iter > iterwin +1 #todo:testen
			lrate = calculate_lrate!(dLL, lrate, mindll, iter,newt_start_iter, do_newton, iterwin)
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
					Lh[i,:] = myAmica.Lt[h,:]
				end
				v[h,:] = 1 ./ sum(exp.(myAmica.Lt-Lh),dims=1)
				vsum[h] = sum(v[h,:])
				myAmica.proportions[h] = vsum[h] / N
				
				if myAmica.proportions[h] == 0
					continue #das continue ist der grund wieso es außerhalb der funktion ist
				end
			end
			try
				myAmica, g, vsum, kappa, lambda = update_parameters_and_other_stuff!(myAmica, v, vsum, h, fp, lambda, rholrate,update_rho)
            catch e
				isa(e,AmicaProportionsZeroException) ? continue : rethrow()
			end
			

			if any(isnan, kappa) || any(isnan, myAmica.source_signals) || any(isnan, lambda) || any(isnan, g) || any(isnan, myAmica.learnedParameters.α)
				println("NaN detected. Better stop.")
				@goto escape_from_NaN
			end
			#Newton
			myAmica = newton_method(myAmica, v, vsum, h, iter, g, kappa, do_newton, newt_start_iter, lrate, lambda)
		end
	
		myAmica = reparameterize!(myAmica, x, v)

		#@show A
		#@show LL[iter]
		show_progress && ProgressMeter.next!(prog; showvalues=[(:LL, myAmica.LL[iter])])
 
	end

    @label escape_from_NaN

	for h in 1:M
		if M > 1
			myAmica.centers[:,h] = myAmica.centers[:,h] + mn #add mean back to model centers
		end
	end

	return myAmica
end


# #calculate z (which is u at first) lines 202 - 218
				#if M > 1 && m > 1
function calculate_z_y!(myAmica,h)
	n = myAmica.n
	m = myAmica.m

    for i in 1:n
        for j in 1:m
            myAmica.y[i,:,j,h] = sqrt(myAmica.learnedParameters.β[j,i,h]) * (myAmica.source_signals[i,:,h] .- myAmica.learnedParameters.μ[j,i,h])
            myAmica.Q[j,:] .= log(myAmica.learnedParameters.α[j,i,h]) + 0.5*log(myAmica.learnedParameters.β[j,i,h]) .+ logpfun(myAmica.y[i,:,j,h],myAmica.learnedParameters.ρ[j,i,h])
        end
        if m > 1
            #hier ist eig. noch berechnung von Qmax und Lt
            for j in 1:m
                Qj = ones(m,1) .* myAmica.Q[j,:]'
                myAmica.z[i,:,j,h] = 1 ./ sum(exp.(myAmica.Q-Qj),dims = 1)
            end
        end
    end
    return myAmica
end


function newton_method(myAmica, v, vsum, h, iter, g, kappa, do_newton, newt_start_iter, lrate::LearningRate, lambda)
	
	lnatrate = lrate.natural_rate
	lrate = lrate.lrate
	M = myAmica.M
	n = myAmica.n
	N = myAmica.N

	if M > 1
		sigma2 = myAmica.source_signals[:,:,h].^2 * v[h,:] /vsum[h]
	else
		sigma2 = sum(myAmica.source_signals.^2,dims=2) / N
	end
	dA = Matrix{Float64}(I, n, n) - g * myAmica.source_signals[:,:,h]' 
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
		myAmica.A[:,:,h] = myAmica.A[:,:,h] + lrate * myAmica.A[:,:,h] * B
	else
		myAmica.A[:,:,h] = myAmica.A[:,:,h] - lnatrate * myAmica.A[:,:,h] * dA
	end
	return myAmica
end


function reparameterize!(myAmica, x, v)
	n = myAmica.n
	M = myAmica.M
	mu = myAmica.learnedParameters.μ
	beta = myAmica.learnedParameters.β

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
			cnew = x * v[h,:] /(sum(v[h,:]))
			for i in 1:n
				Wh = pinv(myAmica.A[:,:,h])
				mu[:,i,h] = mu[:,i,h] .- Wh[i,:]' * (cnew-myAmica.centers[:,h])
			end
			myAmica.centers[:,h] = cnew
		end
	end

	myAmica.learnedParameters.μ .= mu
	myAmica.learnedParameters.β .= beta

	return myAmica
end


function update_parameters_and_other_stuff!(myAmica, v, vsum, h, fp, lambda, lrate_rho::LearningRate, update_rho)
	#it doesnt need iter, just there for test purposes
	rhomin, rhomax, rholrate = lrate_rho.minimum, lrate_rho.maximum, lrate_rho.lrate

	M = myAmica.M
	N = myAmica.N
	n = myAmica.n 
	m = myAmica.m
	alpha = myAmica.learnedParameters.α
	beta = myAmica.learnedParameters.β
	mu = myAmica.learnedParameters.μ
	rho = myAmica.learnedParameters.ρ

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
			else #todo: noch überprüfen, tritt bei erster iteration nicht ein
				if (m > 1) | (M > 1)
					if kp > 0
						mu[j,i,h] = mu[j,i,h] + sqrt(beta[j,i,h]) * sum(zfp[j,:]) / kp #only difference is sqrt instead of 1/sqrt
					end
				end
				db = (rho[j,i,h] * sum(myAmica.z[i,:,j,h].*abs.(myAmica.y[i,:,j,h]).^rho[j,i,h]))^(-2 / rho[j,i,h])
				beta[j,i,h] = beta[j,i,h] * db
			end
	
			if update_rho == 1
				ytmp = abs.(myAmica.y[i,:,j,h]).^rho[j,i,h]
				dr = sum(myAmica.z[i,:,j,h].*log.(ytmp).*ytmp)
	
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

	myAmica.learnedParameters.α = alpha
	myAmica.learnedParameters.β = beta
	myAmica.learnedParameters.μ = mu
	myAmica.learnedParameters.ρ = rho

	return myAmica, g, vsum, kappa, lambda
end