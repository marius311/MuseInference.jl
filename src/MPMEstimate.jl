module MPMEstimate

using Distributions
using ForwardDiff
using Optim
using ProgressMeter
using Random
using Requires
using Setfield

export MPMProblem, mpm


include("mpm.jl")
include("ad.jl")
include("manual.jl")
@init @require Turing="fce5fe82-541a-59a6-adf8-730c64b5f9a0" include("turing.jl")

end