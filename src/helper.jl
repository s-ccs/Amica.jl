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
function logpfun(x,rho)
	return @inbounds -AppleAccelerate.pow(abs.(x), repeat([rho], length(x))) .- log(2) .- loggamma.(1 + 1 / rho)
end


#taken from amica_a.m
function ffun(x::AbstractArray{T, 1}, rho::T) where {T<:Real}
	return @inbounds rho .* sign.(x) .* AppleAccelerate.pow(abs.(x), repeat([rho - 1], length(x)))
end