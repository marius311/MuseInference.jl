
import .Soss
import .Soss.TransformVariables as TV

export SossMuseProblem

struct SossMuseProblem{A<:AD.AbstractBackend, M<:Soss.AbstractModel, MP<:Soss.AbstractModel} <: AbstractMuseProblem
    autodiff :: A
    model :: M
    model_for_prior :: MP
    xform_z
    xform_θ
    x
    observed_vars
    latent_vars
    hyper_vars
end

function SossMuseProblem(
    model::Soss.ConditionalModel; 
    params = (:θ,), 
    autodiff = ForwardDiffBackend()
)
    x = model.obs
    !isempty(x) || error("Model must be conditioned on observed data.")
    sim = rand(model)
    observed_vars = keys(x)
    hyper_vars = (((k, eltype(sim[k])) for k in keys(sim) if k in params)...,)
    latent_vars = keys(delete(sim, (observed_vars..., first.(hyper_vars)...)))
    model_for_prior = Soss.likelihood(Soss.Model(model), first.(hyper_vars)...)(Soss.argvals(model))
    xform_z = Soss.xform(model | select(sim, first.(hyper_vars)))
    xform_θ = Soss.xform(model | select(sim, latent_vars))
    SossMuseProblem(
        autodiff,
        model,
        model_for_prior,
        xform_z,
        xform_θ,        
        x,
        observed_vars,
        latent_vars,
        hyper_vars
    )
end

function transform_θ(prob::SossMuseProblem, θ)
    TV.inverse(prob.xform_θ, _namedtuple(θ))
end

function inv_transform_θ(prob::SossMuseProblem, θ)
    ComponentVector(TV.transform(prob.xform_θ, θ))
end

function logPriorθ(prob::SossMuseProblem, θ::ComponentVector, ::UnTransformedθ)
    Soss.logdensity(prob.model_for_prior(_namedtuple(θ)))
end
function logPriorθ(prob::SossMuseProblem, θ::AbstractVector, ::Transformedθ)
    logPriorθ(prob, inv_transform_θ(prob, θ), UnTransformedθ())
end

function ∇θ_logLike(prob::SossMuseProblem, x, z::AbstractVector, θ::ComponentVector, ::UnTransformedθ)
    like = prob.model | (;x..., TV.transform(prob.xform_z, z)...)
    first(AD.gradient(prob.autodiff, θ -> Soss.logdensity(like, _namedtuple(θ)), θ))
end
function ∇θ_logLike(prob::SossMuseProblem, x, z::AbstractVector, θ::AbstractVector, ::Transformedθ)
    like = prob.model | (;x..., TV.transform(prob.xform_z, z)...)
    first(AD.gradient(prob.autodiff, θ -> Soss.logdensity(like, _namedtuple(inv_transform_θ(prob, θ))), θ))
end


function logLike_and_∇z_logLike(prob::SossMuseProblem, x, z, θ)
    first.(AD.value_and_gradient(prob.autodiff, z -> Soss.logdensity(prob.model | (;x..., _namedtuple(θ)...), TV.transform(prob.xform_z, z)), z))
end

function sample_x_z(prob::SossMuseProblem, rng::AbstractRNG, θ)
    sim = Soss.predict(rng, prob.model, _namedtuple(θ))
    x = select(sim, prob.observed_vars)
    z = TV.inverse(prob.xform_z, select(sim, prob.latent_vars))
    (;x, z)
end

# all user-provided θ pass through this first so get it right type /
# order so the user can be lazy
function standardizeθ(prob::SossMuseProblem, θ::NamedTuple)
    Set(keys(θ)) == Set(first.(prob.hyper_vars)) || error("Expected θ to have keys: $(first.(prob.hyper_vars)).")
    ComponentVector((;(k => T.(θ[k]) for (k,T) in prob.hyper_vars)...))
end
function standardizeθ(prob::SossMuseProblem, θ::Number)
    standardizeθ(prob, (;(prob.hyper_vars[1][1] => θ)))
end

_params_from_θ₀(θ₀::Number) = (:θ,)
_params_from_θ₀(θ₀) = keys(θ₀)

function muse!(result::MuseResult, model::Soss.ConditionalModel, θ₀ = result.θ; kwargs...)
    muse!(result, SossMuseProblem(model, params=_params_from_θ₀(θ₀)), θ₀; kwargs...)
end
function get_J!(result::MuseResult, model::Soss.ConditionalModel, θ₀ = result.θ; kwargs...)
    get_J!(result, SossMuseProblem(model, params=_params_from_θ₀(θ₀)), θ₀; kwargs...)
end
function get_H!(result::MuseResult, model::Soss.ConditionalModel, θ₀ = result.θ; kwargs...)
    get_H!(result, SossMuseProblem(model, params=_params_from_θ₀(θ₀)), θ₀; kwargs...)
end