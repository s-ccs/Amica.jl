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

function ffun!(fp::AbstractArray{T, 1}, x::AbstractArray{T, 1}, rho::T) where {T <: Real}
	optimized_pow!(fp, abs.(x), rho - 1)
	fp .*= sign.(x) .* rho
end

# intelvectormath Pow
function optimized_pow(lhs::AbstractArray{T, 1}, rhs::T)::AbstractArray{T, 1} where {T <: Real}
	out = similar(lhs)
	optimized_pow!(out, lhs, rhs)
	return out
end

function optimized_pow!(out::AbstractArray{Float64, 1}, lhs::AbstractArray{Float64, 1}, rhs::Float64)
	@ccall MKL_jll.libmkl_rt.vdPowx(length(lhs)::Cint, lhs::Ref{Float64}, rhs::Float64, out::Ref{Float64})::Cvoid	
end

function optimized_pow!(out::AbstractArray{Float32, 1}, lhs::AbstractArray{Float32, 1}, rhs::Float32)
	@ccall MKL_jll.libmkl_rt.vsPowx(length(lhs)::Cint, lhs::Ref{Float32}, rhs::Float32, out::Ref{Float32})::Cvoid	
end

# intelvectormath Log

function optimized_log(in::AbstractArray{T})::AbstractArray{T} where {T <: Real}
	out = similar(in)
	optimized_log!(out, in)
	return out
end

function optimized_log!(inout::AbstractArray{T}) where {T <: Real}
	optimized_log!(inout, inout)
end

function optimized_log!(out::AbstractArray{Float64}, in::AbstractArray{Float64})
	@ccall MKL_jll.libmkl_rt.vdLn(length(in)::Cint, in::Ref{Float64}, out::Ref{Float64})::Cvoid	
end

function optimized_log!(out::AbstractArray{Float32}, in::AbstractArray{Float32})
	@ccall MKL_jll.libmkl_rt.vsLn(length(in)::Cint, in::Ref{Float32}, out::Ref{Float32})::Cvoid	
end

# intelvectormath Exp

function optimized_exp(in::AbstractArray{T})::AbstractArray{T} where {T <: Real}
	out = similar(in)
	optimized_exp!(out, in)
	return out
end

function optimized_exp!(inout::AbstractArray{T}) where {T <: Real}
	optimized_exp!(inout, inout)
end

function optimized_exp!(out::AbstractArray{Float64}, in::AbstractArray{Float64})
	@ccall MKL_jll.libmkl_rt.vdExp(length(in)::Cint, in::Ref{Float64}, out::Ref{Float64})::Cvoid	
end

function optimized_exp!(out::AbstractArray{Float32}, in::AbstractArray{Float32})
	@ccall MKL_jll.libmkl_rt.vsExp(length(in)::Cint, in::Ref{Float32}, out::Ref{Float32})::Cvoid	
end