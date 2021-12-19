
### Turing interface

import .Turing
using .Turing: TypedVarInfo, tonamedtuple, decondition, logprior, logjoint
using .Turing.DynamicPPL: evaluate!!

export TuringMuseProblem

struct TuringMuseProblem{A<:AD.AbstractBackend, M, MP, X} <: AbstractMuseProblem
    ad :: A
    model :: M
    model_for_prior :: MP
    x :: X
    function TuringMuseProblem(model, ad::A=AD.ForwardDiffBackend()) where {A<:AD.AbstractBackend}
        x = model.context.values.x
        model = decondition(model)
        vars = tonamedtuple(TypedVarInfo(evaluate!!(model)[2]))
        model_for_prior = model | (x=zero(vars.x[1][1]), z=zero(vars.z[1][1]))
        new{A, typeof(model), typeof(model_for_prior), typeof(x)}(ad, model, model_for_prior, x)
    end
end


muse!(result::MuseResult, model::Turing.Model, args...; kwargs...) = muse!(result, TuringMuseProblem(model), args...; kwargs...)
get_J!(result::MuseResult, model::Turing.Model, args...; kwargs...) = get_J!(result, TuringMuseProblem(model), args...; kwargs...)
get_H!(result::MuseResult, model::Turing.Model, args...; kwargs...) = get_H!(result, TuringMuseProblem(model), args...; kwargs...)


function logPriorθ(prob::TuringMuseProblem, θ)
    logprior(prob.model_for_prior, (;θ))
end

function ∇θ_logLike(prob::TuringMuseProblem, x, θ, z)
    first(AD.gradient(prob.ad, θ -> logjoint(prob.model, (;x, θ, z)), θ))
end

function logLike_and_∇z_logLike(prob::TuringMuseProblem, x, θ, z)
    first.(AD.value_and_gradient(prob.ad, z -> logjoint(prob.model, (;x, θ, z)), z))
end

function sample_x_z(prob::TuringMuseProblem, rng::AbstractRNG, θ)
    model = (prob.model | (;θ))
    vars = tonamedtuple(TypedVarInfo(evaluate!!(model,rng)[2]))
    (;x=vars.x[1][1], z=vars.z[1][1])
end
