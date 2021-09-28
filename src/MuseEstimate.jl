module MuseEstimate

using Base.Iterators: peel, repeated
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

export MuseProblem, muse

include("util.jl")
include("progress.jl")
include("muse.jl")
include("ad.jl")
include("manual.jl")
include("rankreduced.jl")
@init @require Turing="fce5fe82-541a-59a6-adf8-730c64b5f9a0" include("turing.jl")

end