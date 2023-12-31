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

function ffun(x::AbstractArray{T, 3}, rho::AbstractArray{T, 2})::AbstractArray{T, 3} where {T <: Real}
	fp = abs.(x)
	m, n = size(x)

	for j in 1:m 
		for i in 1:n
			@views optimized_pow!(fp[j, i, :], fp[j, i, :], rho[j, i] - 1)
		end
	end

	fp .*= sign.(x) .* rho

	return fp
end

function ffun!(fp::AbstractArray{T, 3}, x::AbstractArray{T, 3}, rho::AbstractArray{T, 2}) where {T <: Real}
	abs_x = abs.(x)
	m, n = size(x)

	for j in 1:m 
		for i in 1:n
			@views optimized_pow!(fp[j, i, :], abs_x[j, i, :], rho[j, i] - 1)
		end
	end

	fp .*= sign.(x) .* rho
end

# intelvectormath Pow

function optimized_pow(lhs::AbstractArray{T, 1}, rhs::T)::AbstractArray{T, 1} where {T <: Real}
	out = similar(lhs)
	optimized_pow!(out, lhs, rhs)
	return out
end

function optimized_pow!(out::AbstractArray{Float32}, lhs::AbstractArray{Float32}, rhs::Float32)

	if !hasproperty(MKL_jll, :libmkl_rt) 
		out .= lhs .^ rhs
		return
	end

	sta = IVM.stride1(lhs)
	sto = IVM.stride1(out)
	dense = (sta == 1 && sto == 1) 

	if dense
		@ccall MKL_jll.libmkl_rt.vsPowx(length(lhs)::Cint, lhs::Ptr{Float32}, rhs::Float32, out::Ptr{Float32})::Cvoid
	else
		@ccall MKL_jll.libmkl_rt.vsPowxI(length(lhs)::Cint, lhs::Ptr{Float32}, sta::Cint, rhs::Float32, out::Ptr{Float32}, sto::Cint)::Cvoid
	end
end

function optimized_pow!(out::AbstractArray{Float64}, lhs::AbstractArray{Float64}, rhs::Float64)
	if !hasproperty(MKL_jll, :libmkl_rt) 
		out .= lhs .^ rhs
		return
	end

	sta = IVM.stride1(lhs)
	sto = IVM.stride1(out)
	dense = (sta == 1 && sto == 1) 

	if dense
		@ccall MKL_jll.libmkl_rt.vdPowx(length(lhs)::Cint, lhs::Ptr{Float64}, rhs::Float64, out::Ptr{Float64})::Cvoid
	else
		@ccall MKL_jll.libmkl_rt.vdPowxI(length(lhs)::Cint, lhs::Ptr{Float64}, sta::Cint, rhs::Float64, out::Ptr{Float64}, sto::Cint)::Cvoid
	end
end

# intelvectormath Log

function optimized_log(in::AbstractArray{T})::AbstractArray{T} where {T <: Real}
	if !hasproperty(MKL_jll, :libmkl_rt) 
		return log.(in)
	end

	return IVM.log(in)
end

function optimized_log!(inout::AbstractArray{T}) where {T <: Real}
	if !hasproperty(MKL_jll, :libmkl_rt) 
		inout .= log.(inout)
		return
	end
	IVM.log!(inout)
end

function optimized_log!(out::AbstractArray{T}, in::AbstractArray{T}) where {T <: Real}
	if !hasproperty(MKL_jll, :libmkl_rt) 
		out .= log.(in)
		return
	end

	IVM.log!(out, in)
end


# intelvectormath Exp

function optimized_exp(in::AbstractArray{T})::AbstractArray{T} where {T <: Real}
	if !hasproperty(MKL_jll, :libmkl_rt) 
		return exp.(in)
	end

	return IVM.exp(in)
end

function optimized_exp!(inout::AbstractArray{T}) where {T <: Real}
	if !hasproperty(MKL_jll, :libmkl_rt) 
		inout .= exp.(inout)
		return
	end

	IVM.exp!(inout)
end

function optimized_exp!(out::AbstractArray{T}, in::AbstractArray{T}) where {T <: Real}
	if !hasproperty(MKL_jll, :libmkl_rt) 
		out .= exp.(in)
		return
	end

	IVM.exp!(out, in)
end
