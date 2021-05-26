
# ensure in right directory and environment
cd(dirname(@__FILE__))
using Pkg
Pkg.activate(".")

using Documenter
using MPMEstimate

makedocs(
    sitename = "MPMEstimate",
    format = Documenter.HTML(
        assets = ["assets/mpmestimate.css"],
        disable_git = true,
    ),
    strict = true,
    modules = [MPMEstimate]
)

deploydocs(
    repo = "https://github.com/marius311/MPMEstimate.jl",
    devbranch = "main",
    devurl = "latest",
    push_preview = true,
    forcepush = true
)
