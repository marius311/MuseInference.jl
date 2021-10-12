
### Turing interface

import .Turing
using .Turing: VarInfo, gradient_logp, SampleFromPrior, DefaultContext

export TuringMuseProblem

struct TuringMuseProblem{M,P} <: AbstractMuseProblem
    model :: M
    model_for_sampling_prior :: P
end

function TuringMuseProblem(model)
    TuringMuseProblem(model, @set(model.args = map(_->missing,model.args)))
end


function muse!(result::MuseResult, model::Turing.Model, θ₀; kwargs...)
    muse!(result, TuringMuseProblem(model), θ₀; kwargs...)
end

function muse!(result::MuseResult, prob::TuringMuseProblem, θ₀; kwargs...)
    muse!(result, prob, prob.model.args.x, θ₀; kwargs...)
end

function get_J!(result::MuseResult, model::Turing.Model, args...; kwargs...)
    get_J!(result, TuringMuseProblem(model), args...; kwargs...)
end

function get_H!(result::MuseResult, model::Turing.Model, args...; kwargs...)
    get_H!(result, TuringMuseProblem(model), args...; kwargs...)
end



# todo: figure out how to use Selector (?) to generically get the
# right parameter sub-spaces rather than manually indexing into these
# vectors

function ∇θ_logLike(prob::TuringMuseProblem, x, θ, z)
    model = prob.model
    @set! model.args.x = x
    θz = [θ; z]
    _, g = gradient_logp(Turing.ForwardDiffAD{min(length(θz),40)}(), θz, VarInfo(model), model)
    θ isa Real ? g[1] : g[1:length(θ)]
end

function logLike_and_∇z_logLike(prob::TuringMuseProblem, x, θ, z)
    model = prob.model
    @set! model.args.x = x
    θz = [θ; z]
    f, g = gradient_logp(Turing.ForwardDiffAD{min(length(θz),40)}(), θz, VarInfo(model), model)
    f, g[length(θ)+1:end]
end

function sample_x_z(prob::TuringMuseProblem, rng::AbstractRNG, θ)
    model = prob.model_for_sampling_prior
    @set! model.args.θ = θ
    vi = VarInfo()
    x = identity.(model(rng, vi))
    z = identity.(vi.metadata.vals[1:(end-length(x))]) 
    (;x, z)
end
