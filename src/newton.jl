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