#removes mean from nxN float matrix
function removeMean!(input)
	mn = mean(input,dims=2)
	(n,N) = size(input)
	for i in 1:n
		input[i,:] = input[i,:] .- mn[i]
	end
	return input
end

#todo:replace with function from lib
function jason_sphering(x)
	(n,N) = size(x)
	Us,Ss,Vs = svd(x*x'/N)
	S = Us * diagm(vec(1 ./sqrt.(Ss))) * Us'
    return x = S*x
end
function bene_sphering(data)
	d_memory_whiten = whitening(data) # Todo: make the dimensionality reduction optional
	return d_memory_whiten.iF * data
end

function calculate_lrate!(dLL, lrateType::LearningRate,mindll, iter, newt_start_iter, do_newton, iterwin)

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
	#myAmica.lrate_over_iterations[iter] = lrate
	return lrateType
end

#no longer in use
function calculate_z_y_Lt!(myAmica,h)
	myAmica.ldet[h] =  -log(abs(det(myAmica.A[:,:,h])))
	myAmica.Lt[h,:] .= log(myAmica.proportions[h]) + myAmica.ldet[h]

	Lt_h = myAmica.Lt[h,:]' #noch nicht übernommen
	n = myAmica.n
	m = myAmica.m
	for i in 1:n
		for j in 1:m
			myAmica.y[i,:,j,h] = sqrt(myAmica.learnedParameters.β[j,i,h]) * (myAmica.source_signals[i,:,h] .- myAmica.learnedParameters.μ[j,i,h])
			myAmica.Q[j,:] .= log(myAmica.learnedParameters.α[j,i,h]) + 0.5*log(myAmica.learnedParameters.β[j,i,h]) .+ logpfun(myAmica.y[i,:,j,h],myAmica.learnedParameters.ρ[j,i,h])
		end
		if m > 1
			Qmax = ones(m,1).*maximum(myAmica.Q,dims=1);
			Lt_h = Lt_h .+ Qmax[1,:]' .+ log.(sum(exp.(myAmica.Q - Qmax),dims = 1))
			for j in 1:m
				Qj = ones(m,1) .* myAmica.Q[j,:]'
				myAmica.z[i,:,j,h] = 1 ./ sum(exp.(myAmica.Q-Qj),dims = 1)
			end
		else
			Lt_h = Lt_h .+ myAmica.Q[1,:]
		end
	end

	myAmica.Lt[h,:] = Lt_h
	return myAmica
end

function calculate_ldet!(myAmica::SingleModelAmica)
	myAmica.ldet = -log(abs(det(myAmica.A)))
end

function calculate_ldet!(myAmica::MultiModelAmica)
	for h in 1:length(myAmica.models)
		myAmica.models[h].ldet = -log(abs(det(myAmica.models[h].A)))
	end
end

function calculate_y!(myAmica::SingleModelAmica)
	Threads.@threads for i in 1:myAmica.n
		for j in 1:myAmica.m
			myAmica.y[i,:,j] = sqrt(myAmica.learnedParameters.scale[j,i]) * (myAmica.source_signals[i,:] .- myAmica.learnedParameters.location[j,i])
		end
	end
end

function calculate_y!(myAmica::MultiModelAmica)
	for h in 1:size(myAmica.models)
		Threads.@threads for i in 1:myAmica.n
			for j in 1:myAmica.m
				myAmica.models[h].y[i,:,j] = sqrt(myAmica.models[h].learnedParameters.scale[j,i]) * (myAmica.models[h].source_signals[i,:] .- myAmica.models[h].learnedParameters.location[j,i])
			end
		end
	end
end

function calculate_Q!(myAmica::SingleModelAmica)
	Threads.@threads for i in 1:myAmica.n
		for j in 1:myAmica.m
			myAmica.Q[j,:] = log(myAmica.learnedParameters.prop[j,i]) + 0.5*log(myAmica.learnedParameters.scale[j,i]) .+ logpfun(myAmica.y[i,:,j],myAmica.learnedParameters.shape[j,i])
		end
	end
end

function calculate_Q!(myAmica::MultiModelAmica)
	for h in 1:size(myAmica.models)
		Threads.@threads for i in 1:myAmica.n
			for j in 1:myAmica.m
				myAmica.models[h].Q[j,:] = log(myAmica.models[h].learnedParameters.prop[j,i]) + 0.5*log(myAmica.models[h].learnedParameters.scale[j,i]) .+ logpfun(myAmica.models[h].y[i,:,j],myAmica.models[h].learnedParameters.shape[j,i])
			end
		end
	end
end

#calculates u but saves it into z
function calculate_u!(myAmica::SingleModelAmica)
	for i in 1:myAmica.n #todo: change how n is stored
		if myAmica.m > 1 #same
			for j in 1:myAmica.m
				Qj = ones(myAmica.m,1) .* myAmica.Q[j,:]'
				myAmica.z[i,:,j] = 1 ./ sum(exp.(myAmica.Q-Qj),dims = 1)
			end
		end
	end
end

function calculate_u!(myAmica::MultiModelAmica)
	m = myAmica.models[1].m #todo: change how m is stored
	n = myAmica.models[1].n #same
	for h in 1:size(models)
		for i in 1:n
			if m > 1 
				for j in 1:m
					Qj = ones(m,1) .* myAmica.models[h].Q[j,:]'
					myAmica.myAmica.models[h].z[i,:,j] = 1 ./ sum(exp.(myAmica.models[h].Q-Qj),dims = 1)
				end
			end
		end
	end
end

function calculate_Lt!(myAmica::SingleModelAmica)
	m = myAmica.m
	for i in 1:myAmica.n #todo: check if this loop is necessary and why, see matlab
		if m > 1
			Qmax = ones(m,1).*maximum(myAmica.Q,dims=1);
			myAmica.Lt = vec(myAmica.Lt' .+ Qmax[1,:]' .+ log.(sum(exp.(myAmica.Q - Qmax),dims = 1))) #todo: check if vec necessary (otherwise dimension mismatch, but it wasnt used in previous version??)
		else
			myAmica.Lt = vec(myAmica.Lt .+ myAmica.Q[1,:]) #todo: test and same as above
		end
	end
end

function calculate_Lt!(myAmica::MultiModelAmica)
	m = myAmica.models[1].m
	for i in 1:myAmica.models[1].n #todo: check if this loop is necessary and why, see matlab
		if m > 1
			Qmax = ones(m,1).*maximum(myAmica.models[h].Q,dims=1);
			myAmica.models[h].Lt = myAmica.models[h].Lt' .+ Qmax[1,:]' .+ log.(sum(exp.(myAmica.models[h].Q - Qmax),dims = 1))
		else
			myAmica.Lt = myAmica.models[h].Lt .+ myAmica.models[h].Q[1,:] #todo: test
		end
	end
end

function calculate_Lt(Lt, Q)
	m = size(Q, 1)
	if m > 1
		Qmax = ones(m,1).*maximum(Q,dims=1);
		return Lt' .+ Qmax[1,:]' .+ log.(sum(exp.(Q - Qmax),dims = 1))
	else
		return Lt .+ Q[1,:] #todo: test
	end
end

function lt_x_proportions_rename_pls(myAmica::MultiModelAmica) #todo: rename
	for h in 1:size(models)
		myAmica.models[h].Lt .= log(myAmica.models[h].proportions) + myAmica.models[h].ldet
	end
end

function lt_x_proportions_rename_pls(myAmica::SingleModelAmica) #todo: rename
	myAmica.Lt .= myAmica.ldet
end

function update_sources!(myAmica::SingleModelAmica, data)
	#b = myAmica.source_signals why was this here?
	b = pinv(myAmica.A) * data
	myAmica.source_signals[:,:,:] = b #todo: check why [:,:,:] needed
end

function update_sources!(myAmica::MultiModelAmica, data)
	n = size(myAmica.models[h].A, 1)
	for h in 1:length(myAmica.models)
		b = myAmica.models[h].source_signals
		for i in 1:n 
			Wh = pinv(myAmica.models[h].A)
			myAmica.models[h].b[i,:] = Wh[i,:]' * data .- Wh[i,:]' * myAmica.models[h].centers #todo: why centers here and not in singlemodel function?
		end
		#myAmica.source_signals[:,:,:] = b
	end
end