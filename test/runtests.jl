using Amica
using Test




if  Sys.isapple()
    # for metal, we need to have an apple system ;-)
    @info "Running Metal Integration Suite..."
    include("metal-integration.jl")
end



for (root, dirs, files) in walkdir(@__DIR__)
    for file in files
        if isnothing(match(r"^test-.*\.jl$", file))
            continue
        end
        title = titlecase(replace(splitext(file[6:end])[1], "-" => " "))
        @testset "$title" begin
            include(file)
        end
    end
end

