"""
Main AMICA algorithm

"""
#todo:remove M from header
# function fit(amicaType::Type{T}, data; M = 1, m = 3, maxiter = 500, remove_mean = true, mu = nothing, beta = nothing, A = nothing, kwargs...) where {T<:AbstractAmica}
# 	if remove_mean
# 		removeMean!(data)
# 		#data = jason_sphering(data)
# 		#data = bene_sphering(data)
		
# 		# f = StatsAPI.fit(Whitening, data)
# 		# transform(f, data)
# 	end
# 	amica = T(data; M = M, m = m, maxiter = maxiter, mu = mu, beta = beta, A = A)
# 	fit!(amica, data; kwargs...)
# 	return amica
# end

function fit(amicaType::Type{T}, data; m = 3, maxiter = 500, remove_mean = true, mu = nothing, beta = nothing, A = nothing, kwargs...) where {T<:AbstractAmica}
	if remove_mean
		removeMean!(data)
		#data = jason_sphering(data)
		#data = bene_sphering(data)
		
		# f = StatsAPI.fit(Whitening, data)
		# transform(f, data)
	end
	amica = T(data; m = m, maxiter = maxiter, mu = mu, beta = beta, A = A)
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
		calculate_ldet!(myAmica)
		lt_x_proportions_rename_pls(myAmica) #todo: rename
		calculate_y!(myAmica)
		#Threads.@threads 
		for i in 1:myAmica.n
			calculate_Q!(myAmica,i)
			calculate_u!(myAmica,i)
			calculate_Lt!(myAmica) #Has to be in loop because it uses current Q
		end
		calculate_LL!(myAmica, iter) #todo: check if iter needs to be given
		
		#calculate difference in loglikelihood between iterations
		if iter > 1
			dLL[iter] = myAmica.LL[iter] - myAmica.LL[iter-1]
		end
		if iter > iterwin +1
			calculate_lrate!(dLL, lrate, iter,newt_start_iter, do_newton, iterwin)
			sdll = sum(dLL[iter-iterwin+1:iter])/iterwin #calculates average likelihood change over multiple itertions
			if (sdll > 0) && (sdll < mindll)
				println("LL increase to low. Stop at iteration ", iter)
				break
			end
		end
   
		try
			update_loop!(myAmica, fp, lambda, rholrate, update_rho, iter, kappa, do_newton, newt_start_iter, lrate) #updates parameters and mixing matrix, todo: zeug übergeben was es für die anderen funktionen braucht
		catch e
			if isa(e,AmicaNaNException)
				println("NaN detected. Better stop. Current iteration: ", iter)
				@goto escape_from_NaN
			else 
				rethrow()
			end
		end
		
		reparameterize!(myAmica, data)
		#Shows current progress
		show_progress && ProgressMeter.next!(prog; showvalues=[(:LL, myAmica.LL[iter])])
 
	end

    @label escape_from_NaN #If parameters contain NaNs, the algorithm skips the A update and terminates by jumping here
	add_means_back!(myAmica)
	return myAmica
end