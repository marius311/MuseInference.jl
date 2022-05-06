
# ensure in right directory and environment
cd(dirname(@__FILE__))
using Pkg
if Pkg.project().name == "MuseInference" 
    using TestEnv # you'll need TestEnv in your global environment
    TestEnv.activate()
end

using Documenter, MuseInference, Turing

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
    strict = true,
    modules = [MuseInference]
)

deploydocs(
    repo = "https://github.com/marius311/MuseInference.jl",
    devbranch = "main",
    devurl = "latest",
    push_preview = true,
    forcepush = true
)
