#removes mean from nxN float matrix
function removeMean!(input)
    mn = mean(input, dims=2)
    (n, N) = size(input)
    for i in 1:n
        input[i, :] .= input[i, :] .- mn[i]
    end
    return mn
end

#Returns sphered data x. todo:replace with function from lib
function sphering_manual!(x::T) where {T}
    (_, N) = size(x)
    Us, Ss = svd(x * x' / N)
    #@debug typeof(Us), typeof(diagm(1 ./ sqrt.(Ss)))
    S = Us * T(diagm((1 ./ sqrt.(Ss)))) * Us'
    x .= S * x
    # @show x
    return S
end

function sphering!(data)
    d_memory_whiten = whitening(data; simple=true)
    data .= d_memory_whiten.iF * data
    return d_memory_whiten
end

#Adds means back to model centers
add_means_back!(myAmica::AbstractAmica, removed_mean) = nothing

function add_means_back!(myAmica::MultiModelAmica, removed_mean)
    M = size(myAmica.models, 1)
    for h in 1:M
        myAmica.models[h].centers = myAmica.models[h].centers + removed_mean #add mean back to model centers
    end
end

#taken from amica_a.m
#L = det(A) * mult p(s|θ)

function logpfun!(out::CuArray{T,3}, y_rho::CuArray{T,3}, shape::CuArray{T,2}) where {T<:Real}
    out .= 1 .+ 1 ./ shape

    out .= .-y_rho .- log(2) .- loggamma.(out)
end
function logpfun!(out::AbstractArray{T,3}, y_rho::AbstractArray{T,3}, shape::AbstractArray{T,2}) where {T<:Real}
    out .= 1 .+ 1 ./ shape
    IVM.lgamma!(out)
    out .= .-y_rho .- log(2) .- out
end

function ffun!(fp::AbstractArray{T,3}, y::AbstractArray{T,3}, rho::AbstractArray{T,2}) where {T<:Real}
    (m, n, N) = size(y)

    fp .= abs.(y)

    for i in 1:n
        for j in 1:m
            @views _fp = fp[j, i, :]
            @views optimized_pow!(_fp, _fp, rho[j, i] - 1)
        end
    end

    fp .*= sign.(y) .* rho
end

# intelvectormath Pow

function optimized_pow(lhs::AbstractArray{T,1}, rhs::T)::AbstractArray{T,1} where {T<:Real}
    out = similar(lhs)
    optimized_pow!(out, lhs, rhs)
    return out

end



function optimized_pow!(out::AbstractArray{Float32}, lhs::AbstractArray{Float32}, rhs::Float32)

    if !hasproperty(MKL_jll, :libmkl_rt) || out isa SubArray{Float32,1,<:CuArray}
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

#function optimized_pow!(out::SubArray{T,1,<:CuArray{T}}, lhs::SubArray{T,1,<:CuArray{T}}, rhs::T) where {T<:Union{Float32,Float64}}
#   out .= lhs .^ rhs
#end
function optimized_pow!(out::AbstractArray{Float64}, lhs::AbstractArray{Float64}, rhs::Float64)
    if !hasproperty(MKL_jll, :libmkl_rt) || out isa SubArray{Float64,1,<:CuArray}
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

function optimized_log(in::CuArray)
    log.(in)
end
function optimized_log(in::AbstractArray{T})::AbstractArray{T} where {T<:Real}
    if !hasproperty(MKL_jll, :libmkl_rt)
        return log.(in)
    end

    return IVM.log(in)
end

function optimized_log!(inout::AbstractArray{T}) where {T<:Real}
    if !hasproperty(MKL_jll, :libmkl_rt)
        inout .= log.(inout)
        return
    end
    IVM.log!(inout)
end

function optimized_log!(out::AbstractArray{T}, in::AbstractArray{T}) where {T<:Real}
    if !hasproperty(MKL_jll, :libmkl_rt)
        out .= log.(in)
        return
    end

    IVM.log!(out, in)
end


# intelvectormath Exp

function optimized_exp(in::AbstractArray{T})::AbstractArray{T} where {T<:Real}
    if !hasproperty(MKL_jll, :libmkl_rt)
        return exp.(in)
    end

    return IVM.exp(in)
end

function optimized_exp!(inout::CuArray)
    inout .= exp.(inout)
end
function optimized_exp!(inout::AbstractArray{T}) where {T<:Real}
    if !hasproperty(MKL_jll, :libmkl_rt)
        inout .= exp.(inout)
        return
    end

    IVM.exp!(inout)
end

function optimized_exp!(out::AbstractArray{T}, in::AbstractArray{T}) where {T<:Real}
    if !hasproperty(MKL_jll, :libmkl_rt)
        out .= exp.(in)
        return
    end

    IVM.exp!(out, in)
end

function optimized_abs!(myAmica::CuSingleModelAmica)
    myAmica.y_rho .= abs.(myAmica.y)
end
function optimized_abs!(myAmica)
    IVM.abs!(myAmica.y_rho, myAmica.y)
end