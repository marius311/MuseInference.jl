
### Turing interface

import .Turing
using .Turing: VarInfo, gradient_logp, SampleFromPrior, DefaultContext

export TuringMPMProblem

struct TuringMPMProblem{M,P} <: AbstractMPMProblem
    model :: M
    model_for_sampling_prior :: P
end

function TuringMPMProblem(model)
    TuringMPMProblem(model, @set(model.args = map(_->missing,model.args)))
end


function mpm(model::Turing.Model, args...; kwargs...)
    mpm(TuringMPMProblem(model), model.args.x, args...; kwargs...)
end


# todo: figure out how to use Selector (?) to generically get the
# right parameter sub-spaces rather than manually indexing into these
# vectors

function ∇θ_logP(prob::TuringMPMProblem, x, θ, z)
    model = prob.model
    @set! model.args.x = x
    θz = [θ; z]
    _, g = gradient_logp(Turing.ForwardDiffAD{min(length(θz),40)}(), θz, VarInfo(model), model)
    θ isa Real ? g[1] : g[1:length(θ)]
end

function logP_and_∇z_logP(prob::TuringMPMProblem, x, θ, z)
    model = prob.model
    @set! model.args.x = x
    θz = [θ; z]
    f, g = gradient_logp(Turing.ForwardDiffAD{min(length(θz),40)}(), θz, VarInfo(model), model)
    f, g[length(θ)+1:end]
end

function sample_x_z(prob::TuringMPMProblem, rng::AbstractRNG, θ)
    model = prob.model_for_sampling_prior
    @set! model.args.θ = θ
    vi = VarInfo()
    x = identity.(model(rng, vi))
    z = identity.(vi.metadata.vals[1:(end-length(x))]) 
    (;x, z)
end
