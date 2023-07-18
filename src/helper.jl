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

function calculate_ldet(A)
	return -log(abs(det(A)))
end

function calculate_y(beta, mu, source_signals)
	return sqrt(beta) * (source_signals .- mu)
end

function calculate_Q(y, alpha, beta, rho)
	return log(alpha) + 0.5*log(beta) .+ logpfun(y,rho)
end

function calculate_u(Q, j)
	m = size(Q, 1)
	Qj = ones(m,1) .* Q[j,:]'
	return 1 ./ sum(exp.(Q-Qj),dims = 1)
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
	
function update_sources!(myAmica::SingleModelAmica, data, h)
	b = myAmica.source_signals
	b = pinv(myAmica.A[:,:,h]) * data
	myAmica.source_signals[:,:,:] = b
	return myAmica
end

function update_sources!(myAmica::MultiModelAmica, data, h)
	b = myAmica.source_signals
	n = myAmica.n
	for i in 1:n 
		Wh = pinv(myAmica.A[:,:,h])
		b[i,:,h] = Wh[i,:]' * data .- Wh[i,:]' * myAmica.centers[:,h]
	end
	myAmica.source_signals[:,:,:] = b
	return myAmica
end