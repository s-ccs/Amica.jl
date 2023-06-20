#removes mean from nxN float matrix
function removeMean(input)
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

	
function get_sources!(myAmica::AbstractAmica,x,h)
	b = myAmica.source_signals
	M = myAmica.M
	n = myAmica.n
	if M == 1
		b = pinv(myAmica.A[:,:,h]) * x
	end
	for i in 1:n 
		if M > 1
			Wh = pinv(myAmica.A[:,:,h])
			b[i,:,h] = Wh[i,:]' * x .- Wh[i,:]' * myAmica.centers[:,h]
		end
	end
	myAmica.source_signals[:,:,:] = b
	return myAmica
end