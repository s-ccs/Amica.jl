mutable struct CuGGParameters{T,ncomps,nmix}
    proportions::CuArray{T,2} #source density mixture proportions
    scale::CuArray{T,2} #source density inverse scale parameter
    location::CuArray{T,2} #source density location parameter
    shape::CuArray{T,2} #source density shape paramters
end




mutable struct CuSingleModelAmica{T,ncomps,nmix} <: AbstractAmica
    source_signals::CuArray{T,2}
    learnedParameters::CuGGParameters{T,ncomps,nmix}
    m::Int    #Number of gaussians
    A::CuArray{T,2} # unmixing matrices for each model
    S::CuArray{T,2} # sphering matrix
    z::CuArray{T,3}
    y::CuArray{T,3}
    centers::CuArray{T,1} #model centers
    Lt::CuArray{T,1} #log likelihood of time point for each model ( M x N )
    LL::CuArray{T,1} #log likelihood over iterations todo: change to tuple 
    ldet::T
    maxiter::Int

    # --- intermediary values
    # precalculated abs(y)^rho
    y_rho::CuArray{T,3}
    lambda::CuArray{T,1}
    fp::CuArray{T,3}
    # z * fp
    zfp::CuArray{T,3}
    g::CuArray{T,2}
    Q::CuArray{T,3}

    u_intermed::CuArray{T,4}
end

myconv(x::GGParameters) = CuArray(x)
myconv(x::AbstractArray) = CuArray(x)
myconv(x) = x
function CuSingleModelAmica(data::AbstractArray{T}; kwargs...) where {T}

    init = SingleModelAmica(data; kwargs...)
    eval(Expr(:call, CuSingleModelAmica,
        [
            :(myconv($init.$field))
            for field in fieldnames(SingleModelAmica)
        ]...))

end


function CUDA.CuArray(gg::Amica.GGParameters{T,n,m}) where {T,n,m}
    CuGGParameters{T,n,m}(
        gg.proportions |> CuArray,
        gg.scale |> CuArray,
        gg.location |> CuArray,
        gg.shape |> CuArray
    )
end