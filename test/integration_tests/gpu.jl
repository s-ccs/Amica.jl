using Revise
using Amica
using BenchmarkTools
using Metal
using LinearAlgebra
using KernelAbstractions
using Atomix

include("util.jl")

data = (read_fdt("input/Memorize.fdt"; ncols=71, T=Float32))';

(N, n) = size(data)
lrate = Amica.LearningRate{Float32}()

# myAmica = SingleModelAmica(Float32, ncomps=n, nsamples=N, m=3, ArrayType=Array);

myAmica = SingleModelAmica(Float32, ncomps=n, nsamples=N, m=3, ArrayType=MtlArray);


Amica.initialize_shape_parameter!(myAmica, lrate)
Amica.removeMean!(data)
S = Amica.sphering!(data)
myAmica.S = S
myAmica.LLdetS = logabsdet(S |> Array)[1]

Amica.update_sources!(myAmica, data)

Amica.calculate_y!(myAmica)
Amica.update_y_rho!(myAmica);

Amica.calculate_u_and_Lt!(myAmica);

@btime Metal.@sync Amica.update_parameters!(myAmica, lrate, true, true);

@kernel function parameters_kernel!(
    sum_zfp::DenseArray{T,2},
    sum_z::DenseArray{T,2},
    dm::DenseArray{T,2},
    dbeta_denom::DenseArray{T,2},
    drho_numer::DenseArray{T,2},
    kp::DenseArray{T,2},
    dlambda_numer::DenseArray{T,2},
    g::DenseArray{T,2},
    @Const(shape::DenseArray{T,2}),
    @Const(y_rho::DenseArray{T,3}),
    @Const(y::DenseArray{T,3}),
    @Const(scale::DenseArray{T,2}),
    @Const(z::DenseArray{T,3}),
    @Const(newton_active::Bool),
) where T<:Real
    k, i, j = @index(Global, NTuple)

    fp = y_rho[k, i, j] * sign(y[k, i, j]) * shape[i, j]

    zfp = z[k, i, j] * fp
    Atomix.@atomic sum_zfp[i, j] += zfp
    Atomix.@atomic sum_z[i, j] += z[k, i, j]

    # kp = sum(z * fp * fp) for use in location update and Newton method
    Atomix.@atomic kp[i, j] += zfp * fp
    Atomix.@atomic dm[i, j] += zfp / y[k, i, j]

    g[k, i] += scale[i, j] * zfp

    if shape[i, j] <= 2
        Atomix.@atomic dbeta_denom[i, j] += zfp * y[k, i, j]
    else
        Atomix.@atomic dbeta_denom[i, j] += z[k, i, j] * y_rho[k, i, j]
    end
    Atomix.@atomic drho_numer[i, j] += z[k, i, j] * log(y_rho[k, i, j]) * y_rho[k, i, j]

    if newton_active
        Atomix.@atomic dlambda_numer[i, j] += z[k, i, j] * (fp * y[k, i, j] - T(1.0))^2
    end

end

backend = KernelAbstractions.get_backend(myAmica.source_signals)
kernel! = parameters_kernel!(backend)

ArrayKind = typeof(myAmica.shape)

N, n, m = size(myAmica.y)
T = Float32


sum_zfp = ArrayKind(zeros(T, n, m))
sum_z = ArrayKind(zeros(T, n, m))
dm = ArrayKind(zeros(T, n, m))
dbeta_denom = ArrayKind(zeros(T, n, m))
drho_numer = ArrayKind(zeros(T, n, m))
kp = ArrayKind(zeros(T, n, m))
dlambda_numer = ArrayKind(zeros(T, n, m))
newton_active = true
# @btime Metal.@sync begin
#     kernel!(
#         sum_zfp,
#         sum_z,
#         dm,
#         dbeta_denom,
#         drho_numer,
#         kp,
#         dlambda_numer,
#         myAmica.g,
#         myAmica.shape,
#         myAmica.y_rho,
#         myAmica.y,
#         myAmica.scale,
#         myAmica.z,
#         newton_active;
#         ndrange=(N, n, m)
#     )
# end



function fp(myAmica::SingleModelAmica)
    myAmica.y_rho .* sign.(myAmica.y) .* Amica.push_dimension(myAmica.shape)
end

@btime Metal.@sync begin
    # sum(z)
    sum_z = sum(myAmica.z, dims=1)[1, :, :]
    # sum(z * fp)
    sum_zfp = sum(myAmica.z .* fp(myAmica), dims=1)[1, :, :]
    # sum(z * fp * fp)
    kp = sum(myAmica.z .* fp(myAmica) .^ 2, dims=1)[1, :, :]
    # sum(z * fp / y)
    dm = sum((myAmica.z .* fp(myAmica)) ./ myAmica.y, dims=1)[1, :, :]
    # sum(z * log(y_rho) * y_rho)
    drho_numer = sum(myAmica.z .* log.(myAmica.y_rho) .* myAmica.y_rho, dims=1)[1, :, :]
    # sum(scale * z * fp)
    myAmica.g = sum(Amica.push_dimension(myAmica.scale) .* myAmica.z .* fp(myAmica), dims=3)[:, :, 1]
    # sum(z * (fp * y - 1)^2)
    dlambda_numer = sum(myAmica.z .* (fp(myAmica) .* myAmica.y .- T(1.0)) .^ 2, dims=1)[1, :, :]
    # myAmica.shape <= 2 ? z * fp * y : z * y_rho
    dbeta_denom = sum(ifelse.(
            Amica.push_dimension(myAmica.shape .<= 2),
            myAmica.z .* fp(myAmica) .* myAmica.y,
            myAmica.z .* myAmica.y_rho), dims=1)[1, :, :]
end