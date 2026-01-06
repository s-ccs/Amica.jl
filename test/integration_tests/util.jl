function build_fortran()
    current_file = @__FILE__
    fortran_dir = joinpath(dirname(current_file), "fortran")
    run(`make -C $fortran_dir`)
end

function run_fortran(config::String, outPath::String)
    current_file = @__FILE__
    script_dir = dirname(current_file)
    amica_exe = joinpath(script_dir, "fortran", "amica")
    full_filename = joinpath(script_dir, config)
    run(setenv(`$amica_exe $full_filename`, "OUT_PATH" => outPath))
end

function read_fdt(path::String; ncols::Int, T::Type=Float32)::Array{T,2}
    file_size = Base.filesize(path)
    nvals = file_size ÷ sizeof(T)
    nrows = nvals ÷ ncols
    data = reinterpret(T, read(path))
    return reshape(data, ncols, nrows)
end

function read_3d_fdt(path::String; ncols::Int, nslabs::Int, T::Type=Float32)::Array{T,3}
    file_size = Base.filesize(path)
    nvals = file_size ÷ sizeof(T)
    nrows = nvals ÷ (ncols * nslabs)
    data = reinterpret(T, read(path))
    return reshape(data, ncols, nrows, nslabs)
end
