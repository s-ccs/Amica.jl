#removes mean from nxN float matrix
function removeMean!(input)
	mn = mean(input,dims=2)
	(n,N) = size(input)
	for i in 1:n
		input[i,:] = input[i,:] .- mn[i]
	end
	return mn
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

function calculate_lrate!(dLL, lrateType::LearningRate, iter, newt_start_iter, do_newton, iterwin) #todo: warum mindll nicht verwendet?

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
	#return lrateType
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
	n = size(myAmica.A,1)
	Threads.@threads for i in 1:n
		for j in 1:myAmica.m
			myAmica.y[i,:,j] = sqrt(myAmica.learnedParameters.scale[j,i]) * (myAmica.source_signals[i,:] .- myAmica.learnedParameters.location[j,i])
		end
	end
end

function calculate_y!(myAmica::MultiModelAmica)
	n = size(myAmica.models[1].A,1)
	for h in 1:size(myAmica.models,1)
		#=Threads.@threads=# for i in 1:n
			for j in 1:myAmica.m
				myAmica.models[h].y[i,:,j] = sqrt(myAmica.models[h].learnedParameters.scale[j,i]) * (myAmica.models[h].source_signals[i,:] .- myAmica.models[h].learnedParameters.location[j,i])
			end
		end
	end
end

#Todo: Rename
function loopiloop(myAmica::SingleModelAmica)
	n = size(myAmica.A,1)

	#Threads.@threads 
	for i in 1:n
		calculate_Q!(myAmica,i)
		calculate_u!(myAmica,i)
		calculate_Lt!(myAmica) #Has to be in loop because it uses current Q
	end
end

function loopiloop(myAmica::MultiModelAmica)
	M = size(myAmica.models,1)
	n = size(myAmica.models[1].A,1)

	for h in 1:M
		for i in 1:n
			calculate_Q!(myAmica,i,h)
			calculate_u!(myAmica,i,h)
			calculate_Lt!(myAmica,h) #Has to be in loop because it uses current Q
		end
	end
end

#calculates u but saves it into z
function calculate_u!(myAmica::SingleModelAmica, i)
	if myAmica.m > 1 #same
		for j in 1:myAmica.m
			Qj = ones(myAmica.m,1) .* myAmica.Q[j,:]'
			myAmica.z[i,:,j] = 1 ./ sum(exp.(myAmica.Q-Qj),dims = 1)
		end
	end
end

function calculate_u!(myAmica::MultiModelAmica,i,h)
	m = myAmica.models[1].m #todo: change how m is stored
	
	if m > 1 
		for j in 1:m
			Qj = ones(m,1) .* myAmica.Q[j,:]'
			myAmica.models[h].z[i,:,j] = 1 ./ sum(exp.(myAmica.Q-Qj),dims = 1)
		end
	end
end
#no longer in use
function calculate_Q!(myAmica::SingleModelAmica, i)
	for j in 1:myAmica.m #todo: change m
		myAmica.Q[j,:] = log(myAmica.learnedParameters.proportions[j,i]) + 0.5*log(myAmica.learnedParameters.scale[j,i]) .+ logpfun(myAmica.y[i,:,j],myAmica.learnedParameters.shape[j,i])
	end
end
#todo: remove n loop
function calculate_Q!(myAmica::MultiModelAmica, i, h)
	for j in 1:myAmica.m #todo: change m
		myAmica.Q[j,:] = log(myAmica.models[h].learnedParameters.proportions[j,i]) + 0.5*log(myAmica.models[h].learnedParameters.scale[j,i]) .+ logpfun(myAmica.models[h].y[i,:,j],myAmica.models[h].learnedParameters.shape[j,i])
	end
end

function calculate_Lt!(myAmica::SingleModelAmica)
	m = myAmica.m
	if m > 1
		Qmax = ones(m,1).*maximum(myAmica.Q,dims=1);
		myAmica.Lt[:] = myAmica.Lt' .+ Qmax[1,:]' .+ log.(sum(exp.(myAmica.Q - Qmax),dims = 1))
	else
		myAmica.Lt[:] = myAmica.Lt .+ myAmica.Q[1,:]#todo: test
	end
end

function calculate_Lt!(myAmica::MultiModelAmica,h)
	m = myAmica.models[1].m

	if m > 1
		Qmax = ones(m,1).*maximum(myAmica.Q,dims=1);
		myAmica.models[h].Lt[:] = myAmica.models[h].Lt[:]' .+ Qmax[1,:]' .+ log.(sum(exp.(myAmica.Q - Qmax),dims = 1))
	else
		myAmica.models[h].Lt[:] = myAmica.models[h].Lt[:] .+ myAmica.Q[1,:] #todo: test
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
	M = size(myAmica.models,1)
	for h in 1:M
		myAmica.models[h].Lt .= log(myAmica.model_proportions[h]) + myAmica.models[h].ldet
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
	n = size(myAmica.models[1].A, 1)
	for h in 1:length(myAmica.models)
		for i in 1:n 
			Wh = pinv(myAmica.models[h].A)
			myAmica.models[h].source_signals[i,:] = Wh[i,:]' * data .- Wh[i,:]' * myAmica.models[h].centers
		end
	end
end

#Adds means back to model centers
add_means_back!(myAmica::SingleModelAmica,removed_mean) = nothing

function add_means_back!(myAmica::MultiModelAmica, removed_mean)
	M = size(myAmica.models,1)
	for h in 1:M
		myAmica.models[h].centers = myAmica.models[h].centers + removed_mean #add mean back to model centers
	end
end

#Sets the initial value for the shape parameter of the GeneralizedGaussians for each Model
function initialize_shape_parameter(myAmica::SingleModelAmica, rholrate::LearningRate)
	myAmica.learnedParameters.shape .= rholrate.init .*myAmica.learnedParameters.shape
end

function initialize_shape_parameter(myAmica::MultiModelAmica, rholrate::LearningRate)
	M = size(myAmica.models,1)
	for h in 1:M
		myAmica.models[h].learnedParameters.shape .= rholrate.init .*myAmica.models[h].learnedParameters.shape
	end
end