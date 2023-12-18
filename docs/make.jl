
# ensure in right directory and environment
cd(dirname(@__FILE__))
using Pkg
if Pkg.project().name == "MuseInference" 
    using TestEnv # you'll need TestEnv in your global environment
    TestEnv.activate()
end

using Documenter, MuseInference, Turing

# uncomment to skip running doc code
# Documenter.Selectors.disable(::Type{Documenter.Expanders.ExampleBlocks}) = true

makedocs(
    sitename = "MuseInference",
    format = Documenter.HTML(
        assets = ["assets/muse.css"],
        disable_git = true,
    ),
    pages = [
        "index.md",
        "userapi.md",
        "devapi.md",
    ],
    checkdocs = :none,
    doctest = false,
    warnonly = false,
    modules = [MuseInference]
)

deploydocs(
    repo = "https://github.com/marius311/MuseInference.jl",
    devbranch = "main",
    devurl = "latest",
    push_preview = true,
    forcepush = true
)
