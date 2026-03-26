module CudaExt

using CUDA, Amica
using PrecompileTools: @setup_workload, @compile_workload

@compile_workload begin
    Amica.fit(
        SingleModelAmica,
        CUDA.zeros(Float32, 3_000, 24),
        m = 3,
        maxiter = 1,
        newt_start_iter = 0,
        show_progress = false,
    )
    Amica.fit(
        SingleModelAmica,
        CUDA.zeros(Float64, 3_000, 24),
        m = 3,
        maxiter = 1,
        newt_start_iter = 0,
        show_progress = false,
    )
end

end
