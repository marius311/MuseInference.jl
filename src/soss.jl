

using .Soss

export SossMuseProblem

struct SossMuseProblem{A<:AD.AbstractBackend, M<:Soss.AbstractModel, MP<:Soss.AbstractModel} <: AbstractMuseProblem
    autodiff :: A
    model :: M
    model_for_prior :: MP
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
    observed_vars = keys(x)
    hyper_vars = params
    latent_vars = keys(delete(rand(model), (observed_vars..., hyper_vars...)))
    model_for_prior = likelihood(Model(model), hyper_vars...)(argvals(model))
    SossMuseProblem(
        autodiff,
        model,
        model_for_prior,
        x,
        observed_vars,
        latent_vars,
        hyper_vars
    )
end

function transform_θ(prob::SossMuseProblem, θ)
    θ # TODO
end

function inv_transform_θ(prob::SossMuseProblem, θ)
    θ # TODO
end

function logPriorθ(prob::SossMuseProblem, θ, θ_space)
    logdensity(prob.model_for_prior(_namedtuple(θ)))
end

function ∇θ_logLike(prob::SossMuseProblem, x, z, θ, θ_space)
    first(AD.gradient(prob.autodiff, θ -> logdensity(prob.model | (;_namedtuple(x)..., _namedtuple(z)...), _namedtuple(θ)), θ))
end

function logLike_and_∇z_logLike(prob::SossMuseProblem, x, z, θ)
    first.(AD.value_and_gradient(prob.autodiff, z -> logdensity(prob.model | (;_namedtuple(x)..., _namedtuple(θ)...), _namedtuple(z)), z))
end

function sample_x_z(prob::SossMuseProblem, rng::AbstractRNG, θ)
    sim = predict(rng, prob.model, _namedtuple(θ))
    x = ComponentVector(select(sim, prob.observed_vars))
    z = ComponentVector(select(sim, prob.latent_vars))
    (;x, z)
end

standardizeθ(prob::SossMuseProblem, θ::NamedTuple) = 1f0 * ComponentVector(θ) # ensure at least Float32
standardizeθ(prob::SossMuseProblem, θ::Number) = 
    length(prob.hyper_vars) == 1 ? standardizeθ(prob, (;θ)) : error("Invalid θ type for this problem.")

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
