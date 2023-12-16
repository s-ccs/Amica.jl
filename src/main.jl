"""
Main AMICA algorithm

"""

function fit(amicaType::Type{T}, data; m = 3, maxiter = 500, location = nothing, scale = nothing, A = nothing, kwargs...) where {T<:AbstractAmica}
	amica = T(data; m = m, maxiter = maxiter, location = location, scale = scale, A = A)
	fit!(amica, data; kwargs...)
	return amica
end
function fit!(amica::AbstractAmica, data; kwargs...)
	amica!(amica, data; kwargs...)
end

function amica!(myAmica::AbstractAmica,
	data;
	lrate = LearningRate(),
	shapelrate = LearningRate(;lrate = 0.1,minimum=0.5,maximum=5,init=1.5),
	remove_mean = true,
	do_sphering = true,
	show_progress = true,
	maxiter = myAmica.maxiter,
	do_newton = 1,
	newt_start_iter = 25,
	iterwin = 10,
	update_shape = 1,
	mindll = 1e-8,

	kwargs...)
	
	initialize_shape_parameter(myAmica,shapelrate)

	(n, N) = size(data)
	m = myAmica.m

	println("m: $(m), n: $(n), N: $(N)")

	#Prepares data by removing means and/or sphering
	if remove_mean
		removed_mean = removeMean!(data)
	end
	if do_sphering
		sphering!(data)
	end
	
	dLL = zeros(1, maxiter)

	#todo put them into object
	lambda = zeros(n, 1)
    prog = ProgressUnknown("Minimizing"; showspeed=true)

	y_rho = zeros(size(myAmica.y))

	for iter in 1:maxiter
		#E-step
		update_sources!(myAmica, data)
		calculate_ldet!(myAmica)
		initialize_Lt!(myAmica)
		calculate_y!(myAmica)
		
		# pre-calculate abs(y)^rho
		for j in 1:m
			for i in 1:n
				y_rho[i,:,j] .= optimized_pow(abs.(myAmica.y[i,:,j]), myAmica.learnedParameters.shape[j,i])
			end
		end



		loopiloop!(myAmica, y_rho) #Updates y and Lt. Todo: Rename
		calculate_LL!(myAmica)
		@debug (:LL,myAmica.LL)
		#Calculate difference in loglikelihood between iterations
		if iter > 1
			dLL[iter] = myAmica.LL[iter] - myAmica.LL[iter-1]
		end
		if iter > iterwin +1
			calculate_lrate!(dLL, lrate, iter,newt_start_iter, do_newton, iterwin)
			#Calculates average likelihood change over multiple itertions
			sdll = sum(dLL[iter-iterwin+1:iter])/iterwin
			#Checks termination criterion
			if (sdll > 0) && (sdll < mindll)
				println("LL increase to low. Stop at iteration ", iter)
				break
			end
		end
		
		#M-step
		try
			#Updates parameters and mixing matrix
			update_loop!(myAmica, lambda, y_rho, shapelrate, update_shape, iter, do_newton, newt_start_iter, lrate)
		catch e
			#Terminates if NaNs are detected in parameters
			if isa(e,AmicaNaNException)
				println("\nNaN detected. Better stop. Current iteration: ", iter)
				@goto escape_from_NaN
			else 
				rethrow()
			end
		end
		
		reparameterize!(myAmica, data)
		#Shows current progress
		show_progress && ProgressMeter.next!(prog; showvalues=[(:LL, myAmica.LL[iter])])
 
	end
	#If parameters contain NaNs, the algorithm skips the A update and terminates by jumping here
    @label escape_from_NaN
	#If means were removed, they are added back
	if remove_mean
		add_means_back!(myAmica, removed_mean)
	end
	return myAmica
end