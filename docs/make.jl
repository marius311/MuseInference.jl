
# ensure in right directory and environment
cd(dirname(@__FILE__))
using Pkg
Pkg.activate(".")

using Documenter
using MuseEstimate

makedocs(
    sitename = "MuseEstimate",
    format = Documenter.HTML(
        assets = ["assets/muse.css"],
        disable_git = true,
    ),
    pages = [
        "index.md",
        "api.md",
    ],
    checkdocs = :none,
    strict = true,
    modules = [MuseEstimate]
)

deploydocs(
    repo = "https://github.com/marius311/MuseEstimate.jl",
    devbranch = "main",
    devurl = "latest",
    push_preview = true,
    forcepush = true
)
