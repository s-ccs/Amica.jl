#removes mean from nxN float matrix
function removeMean!(input)
	mn = mean(input,dims=2)
	(n,N) = size(input)
	for i in 1:n
		input[i,:] .= input[i,:] .- mn[i]
	end
	return mn
end

#Returns sphered data x. todo:replace with function from lib
function sphering!(x)
	(_,N) = size(x)
	Us,Ss = svd(x*x'/N)
	S = Us * diagm(vec(1 ./sqrt.(Ss))) * Us'
    x .= S*x
	return S
end

function bene_sphering(data)
	d_memory_whiten = whitening(data) # Todo: make the dimensionality reduction optional
	return d_memory_whiten.iF * data
end

#Adds means back to model centers
add_means_back!(myAmica::SingleModelAmica,removed_mean) = nothing

function add_means_back!(myAmica::MultiModelAmica, removed_mean)
	M = size(myAmica.models,1)
	for h in 1:M
		myAmica.models[h].centers = myAmica.models[h].centers + removed_mean #add mean back to model centers
	end
end

#taken from amica_a.m
#L = det(A) * mult p(s|θ)
function logpfun(rho, y_rho)
	return .- y_rho .- log(2) .- loggamma(1 + 1 / rho)
end


#taken from amica_a.m
@views function ffun(x::AbstractArray{T, 1}, rho::T) where {T<:Real}
	return @inbounds copysign.(optimized_pow(abs.(x), rho - 1), x) .* rho
end

# optimized power function for different cpu architectures
function optimized_pow(lhs::AbstractArray{T, 1}, rhs::T) where {T<:Real}
	optimized_pow(lhs, repeat([rhs], length(lhs)))
end

function optimized_pow(lhs::AbstractArray{T, 1}, rhs::AbstractArray{T, 1}) where {T<:Real}
#	if Sys.iswindows() || Sys.islinux()
#		return IVM.pow(lhs, rhs)
#	elseif Sys.isapple()
#		return AppleAccelerate.pow(lhs, rhs)
#	else 
		return lhs .^ rhs
#	end
end

function optimized_log(val)
	if Sys.iswindows() || Sys.islinux()
		return IVM.log(val)
	elseif Sys.isapple()
		return AppleAccelerate.log(val)
	else 
		return log.(val)
	end
end


function optimized_exp(val) 
	#if Sys.iswindows() || Sys.islinux()
	#		return IVM.exp(val)
	#elseif Sys.isapple()
#		return AppleAccelerate.exp(val)
	#else 
		return exp.(val)
	#end
end
