module MuseInference

using AbstractDifferentiation
using Base.Iterators: peel, repeated
using ComponentArrays
using CovarianceEstimation
using Dates
using Distributed
using Distributions
using FileIO
using FiniteDifferences
using ForwardDiff
using LinearAlgebra
using Markdown
using NamedTupleTools
using Optim
using Printf
using ProgressMeter
using Random
using Requires
using Setfield
using Statistics
using Test
using UnPack

export MuseProblem, MuseResult, muse, muse!, get_J!, get_H!

include("util.jl")
include("ad.jl")
include("progress.jl")
include("interface.jl")
include("muse.jl")
include("manual.jl")
@init @require Turing="fce5fe82-541a-59a6-adf8-730c64b5f9a0" include("turing.jl")
@init @require Soss="8ce77f84-9b61-11e8-39ff-d17a774bf41c" include("soss.jl")

end