module MPMEstimate

using Base.Iterators: peel
using Distributed
using Distributions
using FiniteDifferences
using ForwardDiff
using LinearAlgebra
using Optim
using ProgressMeter
using Random
using Requires
using Setfield

export MPMProblem, mpm

include("util.jl")
include("progress.jl")
include("mpm.jl")
include("ad.jl")
include("manual.jl")
@init @require Turing="fce5fe82-541a-59a6-adf8-730c64b5f9a0" include("turing.jl")

end