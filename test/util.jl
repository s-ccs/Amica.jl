function integration_test_dir()
    return dirname(@__FILE__)
end

function integration_test_path(parts...)
    return joinpath(integration_test_dir(), parts...)
end

function integration_input_path(parts...)
    return integration_test_path("input", parts...)
end

function integration_dump_dir(tool::Symbol, mode::Symbol)
    mode_name = mode === :without_newton ? "without_newton" : mode === :with_newton ? "with_newton" : error("Unsupported mode: $mode")
    tool_name = tool === :fortran ? "fortran" : tool === :julia ? "julia" : error("Unsupported tool: $tool")
    return integration_test_path("dumps", tool_name, mode_name)
end

function prepare_integration_dump_dirs!()
    mkpath(integration_dump_dir(:fortran, :without_newton))
    mkpath(integration_dump_dir(:fortran, :with_newton))
    mkpath(integration_dump_dir(:julia, :without_newton))
    mkpath(integration_dump_dir(:julia, :with_newton))
    return nothing
end

function cleanup_integration_dumps!()
    for rel_path in ("dumps", "datadumps", "datadumps_newton")
        path = integration_test_path(rel_path)
        if ispath(path)
            rm(path; force=true, recursive=true)
        end
    end
    return nothing
end

function build_fortran()
    fortran_dir = integration_test_path("fortran")
    run(`make -C $fortran_dir`)
end

function run_fortran(config::String, out_path::String)
    script_dir = integration_test_dir()
    amica_exe = integration_test_path("fortran", "amica")
    full_filename = integration_input_path(config)
    resolved_out_path = isabspath(out_path) ? out_path : integration_test_path(out_path)
    mkpath(resolved_out_path)

    base_cmd = `$amica_exe $full_filename`
    cmd = Cmd(base_cmd; dir=script_dir)
    run(setenv(cmd, "OUT_PATH" => resolved_out_path, "OMPI_MCA_plm" => "isolated"))
end

function read_fdt(path::String; ncols::Int, T::Type=Float32, transpose=false, OutType=Float64)::Array{OutType,2}
    file_size = Base.filesize(path)
    file_size % sizeof(T) == 0 || error("File size is not divisible by element size")

    nvals = file_size ÷ sizeof(T)
    nvals % ncols == 0 || error("Number of values is not divisible by ncols")
    nrows = nvals ÷ ncols

    out = transpose ? Matrix{OutType}(undef, nrows, ncols) : Matrix{OutType}(undef, ncols, nrows)

    open(path, "r") do io
        if !transpose && OutType === T
            read!(io, vec(out))
            return out
        end

        rowbuf = Vector{T}(undef, ncols)
        for row in 1:nrows
            read!(io, rowbuf)
            if transpose
                @inbounds @simd for col in 1:ncols
                    out[row, col] = OutType(rowbuf[col])
                end
            else
                @inbounds @simd for col in 1:ncols
                    out[col, row] = OutType(rowbuf[col])
                end
            end
        end
    end

    return out
end

function read_3d_fdt(path::String; ncols::Int, nslabs::Int, T::Type=Float32, transpose=false, OutType=Float64)::Array{OutType,3}
    file_size = Base.filesize(path)
    nvals = file_size ÷ sizeof(T)
    nrows = nvals ÷ (ncols * nslabs)
    data = reinterpret(T, read(path))
    data = reshape(data, ncols, nrows, nslabs)

    if transpose
        data = data'
    end
    return OutType.(data)
end
