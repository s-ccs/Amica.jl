using Amica
using Documenter

DocMeta.setdocmeta!(Amica, :DocTestSetup, :(using Amica); recursive = true)

makedocs(;
    modules = [Amica],
    authors = "Valentin Morlock, Alexander Lulkin, Benedikt V. Ehinger",
    repo = "https://github.com/s-ccs/Amica.jl/blob/{commit}{path}#{line}",
    sitename = "Amica.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://s-ccs.github.io/Amica.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Introduction" => "introduction.md",
        "Getting Started" => "getting_started.md",
        "Configuration" => "configuration.md",
        "Performance" => "performance.md",
        "Contribution?" => [
            "Contribution!" => "90-contributing.md",
            "Developer Docs" => "91-developer.md",
        ],
        "API/Reference" => "95-reference.md",
    ],
)

deploydocs(; repo = "github.com/s-ccs/Amica.jl", devbranch = "main", push_preview = true)
