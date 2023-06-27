#removes mean from nxN float matrix
function removeMean!(input)
	mn = mean(input,dims=2)
	(n,N) = size(input)
	for i in 1:n
		input[i,:] = input[i,:] .- mn[i]
	end
	return input
end


function calculate_lrate!(dLL, lrateType::LearningRate,mindll, iter, newt_start_iter, do_newton, iterwin)

	lratefact,lnatrate,lratemax, = lrateType.decreaseFactor, lrateType.natural_rate, lrateType.maximum
	lrate = lrateType.lrate
	sdll = sum(dLL[iter-iterwin+1:iter])/iterwin
    if (sdll > 0) && (sdll < mindll)
        return -1
    end
    if sdll < 0
        println("Likelihood decreasing!")
        lrate = lrate * lratefact
    else
        #lrate über zeit nochmal anschauen. wird sie größer??
        if (iter > newt_start_iter) && do_newton == 1
            lrate = min(lratemax,lrate + min(0.1,lrate))
        else
            lrate = min(lnatrate,lrate + min(0.1,lrate))
        end
    end
	lrateType.lrate = lrate

	return lrateType
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
			for j in 1:m
				Qj = ones(m,1) .* myAmica.Q[j,:]'
				myAmica.z[i,:,j,h] = 1 ./ sum(exp.(myAmica.Q-Qj),dims = 1)
			end
		end
	end
	return myAmica
end

#attempted fix for Lt calculation (didnt work)
function calculate_z_y_Lt!(myAmica,h)
	myAmica.ldet[h] =  -log(abs(det(myAmica.A[:,:,h])))
	myAmica.Lt[h,:] .= log(myAmica.proportions[h]) + myAmica.ldet[h]

	Lt_h = myAmica.Lt[h]'
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
	
function get_sources!(myAmica::AbstractAmica, data, h)
	b = myAmica.source_signals
	M = myAmica.M
	n = myAmica.n
	if M == 1
		b = pinv(myAmica.A[:,:,h]) * data
	end
	for i in 1:n 
		if M > 1
			Wh = pinv(myAmica.A[:,:,h])
			b[i,:,h] = Wh[i,:]' * data .- Wh[i,:]' * myAmica.centers[:,h]
		end
	end
	myAmica.source_signals[:,:,:] = b
	return myAmica
end