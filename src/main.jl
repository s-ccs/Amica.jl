"""
Main AMICA algorithm

"""
function fit(amicaType::Type{T}, data; M = 1, m = 3, maxiter = 500, remove_mean = true, mu = nothing, beta = nothing, A = nothing, kwargs...) where {T<:AbstractAmica}
	if remove_mean
		removeMean!(data)
		data = jason_sphering(data)
		#data = bene_sphering(data)
		
		# f = StatsAPI.fit(Whitening, data)
		# transform(f, data)
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
	
	myAmica.learnedParameters.shape .= rholrate.init .*myAmica.learnedParameters.shape
	#learnedParameters(m::AbstractAmica) = m.learnedParameters
	

	n = myAmica.n
	N = myAmica.N
	m = myAmica.m


	#Mx = maximum(abs.(data)) #maximum and max are not the same

	mn = mean(data, dims = 2) #should be zeros if remove_mean = 0
	#a = 0
	g = zeros(n, N)
	v = ones(N)
	lambda = zeros(n, 1)
	kappa = zeros(n, 1)
	sigma2 = zeros(n, 1)

	dLL = zeros(1, maxiter)

	fp = zeros(m ,N)

	#r = zeros(n,N,m,M)
	#todo put them into object
	lambda = zeros(n, 1)
	kappa = zeros(n, 1)
	sigma2 = zeros(n, 1)

    prog = ProgressUnknown("Minimizing"; showspeed=true)


	for iter in 1:maxiter
		update_sources!(myAmica, data)
		calculate_ldet!(myAmica) #todo: funktionen auf einzelnde models anwenden seperat anwenden (keine schleife außen)
		lt_x_proportions_rename_pls(myAmica)
		calculate_y!(myAmica)
		calculate_Q!(myAmica)
		calculate_u!(myAmica)
		calculate_Lt!(myAmica) #todo: check weird stuff with position in loop
		calculate_LL!(myAmica, iter) #todo: check if iter needs to be given
		
		if iter > 1
			dLL[iter] = myAmica.LL[iter] - myAmica.LL[iter-1]
		end
		if iter > iterwin +1 #todo:testen
			lrate = calculate_lrate!(dLL, lrate, mindll, iter,newt_start_iter, do_newton, iterwin)
			#lrate < 0 ? break : ""
			sdll = sum(dLL[iter-iterwin+1:iter])/iterwin
           # @show sdll
			if (sdll > 0) && (sdll < mindll)
				println("LL increase to low. Stop at iteration ", iter)
				break
			end
			#println("Iteration: ", iter, ". lrate = ", lrate.lrate, ". LL = ", myAmica.LL[iter])
		end
   
		M = 1 #todo: remove
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
				#myAmica, g, vsum, kappa, lambda = update_parameters_and_other_stuff!(myAmica, v, vsum, h, fp, lambda, rholrate, update_rho)
				myAmica, g, kappa, lambda = update_parameters!(myAmica, v, vsum, fp, lambda, rholrate, update_rho) #should get h for multimodel
            catch e
				isa(e,AmicaProportionsZeroException) ? continue : rethrow()
			end
			

			if any(isnan, kappa) || any(isnan, myAmica.source_signals) || any(isnan, lambda) || any(isnan, g) || any(isnan, myAmica.learnedParameters.prop)
				println("NaN detected. Better stop. Current iteration: ", iter)
				@goto escape_from_NaN
			end
			#Newton
			myAmica = newton_method(myAmica, v, vsum, h, iter, g, kappa, do_newton, newt_start_iter, lrate, lambda)
		end
	
		myAmica = reparameterize!(myAmica, data, v)

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