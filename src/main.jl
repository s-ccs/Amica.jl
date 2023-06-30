"""
Main AMICA algorithm

"""
function fit(amicaType::Type{T}, data; M = 1, m = 3, maxiter = 500, remove_mean = true, mu = nothing, beta = nothing, A = nothing, kwargs...) where {T<:AbstractAmica}
	if remove_mean
		removeMean!(data)
	end
	amica = T(data; M = M, m = m, maxiter = maxiter, mu = mu, beta = beta, A = A)
	fit!(amica, data; kwargs...)
	return amica
end

function fit!(amica::AbstractAmica, data; kwargs...)
	amica!(amica, data; kwargs...)
end

function amica!(myAmica::AbstractAmica,
	data;
	lrate = LearningRate(),
	rholrate = LearningRate(;lrate = 0.1,minimum=0.5,maximum=5,init=1.5),
	dispinterval = 50,
	showLL = 1,
	plotfig = 1,
	plothist = 1,
	nbins = 50,
	show_progress = true,
	maxiter = myAmica.maxiter,
	do_newton = 1,
	newt_start_iter = 25,# TODO Check
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


	#Mx = maximum(abs.(data)) #maximum and max are not the same

	mn = mean(data, dims = 2) #should be zeros if remove_mean = 0
	#a = 0
	g = zeros(n, N)
	v = ones(M, N)
	lambda = zeros(n, 1)
	kappa = zeros(n, 1)
	sigma2 = zeros(n, 1)

	dLL = zeros(1, maxiter)

	fp = zeros(m ,N)

	#r = zeros(n,N,m,M)
	
	lambda = zeros(n, 1)
	kappa = zeros(n, 1)
	sigma2 = zeros(n, 1)

    prog = ProgressUnknown("Minimizing"; showspeed=true)


	for iter in 1:maxiter
		for h in 1:M
			myAmica = get_sources!(myAmica, data, h)
			#Lt[iter,:] = get_likelihood_time(A, proportions, mu, beta, rho, alpha, b, h)
			#Lt[1,:] = [-84.2453 -40.6495 -9.3180 -7.9679 -38.9525 -83.2213]
			#myAmica = calculate_z_y!(myAmica,h)
			#Lt[h,:] = sum(loglikelihoodMMGG.(eachcol(mu[:,:,h]),eachcol(beta[:,:,h]),eachcol(rho[:,:,h]),eachrow(source_signals[:,:,h]),eachcol(alpha[:,:,h])))
			#myAmica = calculate_Lt!(myAmica, h)
			myAmica = calculate_z_y_Lt!(myAmica, h)
			myAmica = calculate_LL!(myAmica, iter)
		end
		
		if iter > 1
			dLL[iter] = myAmica.LL[iter] - myAmica.LL[iter-1]
		end
		if iter > iterwin +1 #todo:testen
			lrate = calculate_lrate!(dLL, lrate, mindll, iter,newt_start_iter, do_newton, iterwin)
			#lrate < 0 ? break : ""
			sdll = sum(dLL[iter-iterwin+1:iter])/iterwin
			
            
			if (sdll > 0) && (sdll < mindll)
				println("LL increase to low. Stop at iteration ", iter)
				break
			end
			#println("Iteration: ", iter, ". lrate = ", lrate.lrate, ". LL = ", myAmica.LL[iter])
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
				myAmica, g, vsum, kappa, lambda = update_parameters_and_other_stuff!(myAmica, v, vsum, h, fp, lambda, rholrate, update_rho)
            catch e
				isa(e,AmicaProportionsZeroException) ? continue : rethrow()
			end
			

			if any(isnan, kappa) || any(isnan, myAmica.source_signals) || any(isnan, lambda) || any(isnan, g) || any(isnan, myAmica.learnedParameters.α)
				println("NaN detected. Better stop. Current iteration: ", iter)
				@goto escape_from_NaN
			end
			#Newton
			myAmica = newton_method(myAmica, v, vsum, h, iter, g, kappa, do_newton, newt_start_iter, lrate, lambda)
		end
	
		myAmica = reparameterize!(myAmica, data, v)

		#@show A
		#@show LL[iter]
		show_progress && ProgressMeter.next!(prog; showvalues=[(:LL, myAmica.LL[iter]),(:lrate, lrate.lrate)])
 
	end

    @label escape_from_NaN

	for h in 1:M
		if M > 1
			myAmica.centers[:,h] = myAmica.centers[:,h] + mn #add mean back to model centers
		end
	end
	return myAmica
end