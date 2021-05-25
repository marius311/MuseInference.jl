
# ensure in right directory and environment
cd(dirname(@__FILE__))
using Pkg
pkg"activate ."

using Documenter
using MPMEstimate

makedocs(
    sitename = "MPMEstimate",
    format = Documenter.HTML(
        assets = ["assets/mpmestimate.css"],
        disable_git = true,
    ),
    modules = [MPMEstimate]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
