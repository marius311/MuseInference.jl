
### Turing interface

import .Turing
using .Turing: TypedVarInfo, tonamedtuple, decondition, logprior, logjoint
using .Turing.DynamicPPL: evaluate!!

export TuringMuseProblem


struct TuringMuseProblem{A<:AD.AbstractBackend, M<:Turing.Model} <: AbstractMuseProblem
    
    autodiff :: A
    model :: M
    model_for_prior
    x
    observed_vars
    latent_vars
    hyper_vars

end

@doc doc"""

    TuringMuseProblem(
        model;
        observed_vars=(:x,), latent_vars=(:z,), hyper_vars=(:θ,), 
        autodiff=MuseEstimate.ForwardDiffBackend()
    )

Wrap a Turing model to be ready to pass to [`muse`](@ref). 

The names of variables within the model which are observed, latent, or
hyperparameters, can be customized via keyword arguments which accept
tuples of names. E.g., 

```julia
@model function demo()
    σ ~ Normal(0, 3)
    z ~ MvNormal(zeros(512), exp(σ/2))
    w ~ MvNormal(z, 1)
    x ~ MvNormal(w, 1)
    y ~ MvNormal(w, 2)
    (;σ,z,w,x,y)
end
truth = demo()()
model = demo() | (;truth.x, truth.y)
prob = TuringMuseProblem(
    model, observed_vars=[:x,:y], latent_vars=[:z,:w], hyper_vars=[:σ]
)
```

!!! note

    When defining Turing models to be used with MuseEstimate, the new-style definition 
    of Turing models is required, where the random variables do not appear as arguments 
    to the function. This is because internally, MuseEstimate needs
    to [`condition`](https://turinglang.github.io/DynamicPPL.jl/stable/#AbstractPPL.condition-Tuple{Model}) 
    your model on various variables.

The `autodiff` parameter should be either
`MuseEstimate.ForwardDiffBackend()` or
``MuseEstimate.ZygoteBackend()`, specifying which library to use for
automatic differenation. 

"""
function TuringMuseProblem(
    model; 
    observed_vars = (:x,),
    latent_vars = (:z,),
    hyper_vars = (:θ,),
    autodiff = AD.ForwardDiffBackend()
)

    x = ComponentVector(select(model.context.values, observed_vars))
    model = decondition(model)
    vars = map(first∘first, tonamedtuple(TypedVarInfo(evaluate!!(model)[2])))
    model_for_prior = model | map(zero, select(vars, observed_vars)) | map(zero, select(vars, latent_vars))
    TuringMuseProblem(autodiff, model, model_for_prior, x, observed_vars, latent_vars, hyper_vars)

end

standardizeθ(prob::TuringMuseProblem, θ::NamedTuple) = ComponentVector(θ)
standardizeθ(prob::TuringMuseProblem, θ::Number) = length(prob.hyper_vars) == 1 ? ComponentVector(;θ) : error("Invalid θ type for this problem.")

function logPriorθ(prob::TuringMuseProblem, θ)
    logprior(prob.model_for_prior, θ)
end

function ∇θ_logLike(prob::TuringMuseProblem, x, z, θ)
    first(AD.gradient(prob.autodiff, θ -> logjoint(prob.model, (;NamedTupleView(x)..., NamedTupleView(z)..., NamedTupleView(θ)...)), θ))
end

function logLike_and_∇z_logLike(prob::TuringMuseProblem, x, z, θ)
    first.(AD.value_and_gradient(prob.autodiff, z -> logjoint(prob.model, (;NamedTupleView(x)..., NamedTupleView(z)..., NamedTupleView(θ)...)), z))
end

function sample_x_z(prob::TuringMuseProblem, rng::AbstractRNG, θ)
    model = (prob.model | NamedTupleView(θ))
    vars = map(first∘first, tonamedtuple(TypedVarInfo(evaluate!!(model,rng)[2])))
    (;x=ComponentVector(select(vars, prob.observed_vars)), z=ComponentVector(select(vars, prob.latent_vars)))
end


muse!(result::MuseResult, model::Turing.Model, args...; kwargs...) = muse!(result, TuringMuseProblem(model), args...; kwargs...)
get_J!(result::MuseResult, model::Turing.Model, args...; kwargs...) = get_J!(result, TuringMuseProblem(model), args...; kwargs...)
get_H!(result::MuseResult, model::Turing.Model, args...; kwargs...) = get_H!(result, TuringMuseProblem(model), args...; kwargs...)
