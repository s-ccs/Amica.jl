function newton_method!(myAmica::SingleModelAmica, iter, g, kappa, do_newton, newt_start_iter, lrate::LearningRate, lambda)
	
	lnatrate = lrate.natural_rate
	lrate = lrate.lrate
	n = myAmica.n
	N = myAmica.N

	sigma2 = sum(myAmica.source_signals.^2,dims=2) / N

	dA = Matrix{Float64}(I, n, n) - g * myAmica.source_signals[:,:]' 
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
		myAmica.A[:,:] = myAmica.A[:,:] + lrate * myAmica.A[:,:] * B
	else
		myAmica.A[:,:] = myAmica.A[:,:] - lnatrate * myAmica.A[:,:] * dA
	end
end

function newton_method!(myAmica::MultiModelAmica, v, vsum, h, iter, g, kappa, do_newton, newt_start_iter, lrate::LearningRate, lambda)
	
	lnatrate = lrate.natural_rate
	lrate = lrate.lrate
	M = size(myAmica.models)
	n = myAmica.n
	N = myAmica.N

	sigma2 = myAmica.source_signals[:,:,h].^2 * v[h,:] /vsum[h]

	dA = Matrix{Float64}(I, n, n) - g * myAmica.models[h].source_signals[:,:]' 
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
		myAmica.models[h].A[:,:] = myAmica.models[h].A[:,:] + lrate * myAmica.models[h].A[:,:] * B
	else
		myAmica.models[h].A[:,:] = myAmica.models[h].A[:,:] - lnatrate * myAmica.models[h].A[:,:] * dA
	end
end