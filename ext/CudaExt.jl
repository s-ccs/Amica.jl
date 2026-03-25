module CudaExt

using CUDA, Amica
using PrecompileTools: @setup_workload, @compile_workload

@compile_workload begin
    Amica.fit(
        SingleModelAmica,
        zeros(Float32, 3_000, 24),
        m = 3,
        maxiter = 1,
        newt_start_iter = 0,
        show_progress = false,
        ArrayType = CuArray,
    )
    Amica.fit(
        SingleModelAmica,
        zeros(Float64, 3_000, 24),
        m = 3,
        maxiter = 1,
        newt_start_iter = 0,
        show_progress = false,
        ArrayType = CuArray,
    )
end

end
