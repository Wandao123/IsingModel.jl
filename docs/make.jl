using IsingModel
using Documenter

DocMeta.setdocmeta!(IsingModel, :DocTestSetup, :(using IsingModel); recursive=true)

makedocs(;
    modules=[IsingModel],
    authors="Wandao123 <20665675+Wandao123@users.noreply.github.com>",
    repo="https://github.com/Wandao123/IsingModel.jl/blob/{commit}{path}#{line}",
    sitename="IsingModel.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Wandao123.github.io/IsingModel.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Wandao123/IsingModel.jl",
    devbranch="main",
)
