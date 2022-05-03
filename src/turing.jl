
### Turing interface

import .Turing
using .Turing: VarInfo, TypedVarInfo, tonamedtuple, decondition, logprior, 
    logjoint, VarName, SampleFromPrior, link!, invlink!, MAP, OptimLogDensity, 
    DefaultContext, SimpleVarInfo, LikelihoodContext, condition
using .Turing.ModeEstimation: transform!
using .Turing.DynamicPPL: evaluate!!, setval!, settrans!, Model, Metadata, getlogp

import .Turing.DynamicPPL: VarInfo, condition

export TuringMuseProblem


struct TuringMuseProblem{A<:AD.AbstractBackend, M<:Turing.Model} <: AbstractMuseProblem
    
    autodiff :: A
    model :: M
    model_for_prior
    vi_z_θ
    vi_θ
    x
    observed_vars
    latent_vars
    hyper_vars

end

@doc doc"""

    TuringMuseProblem(
        model;
        observed_vars=(:x,), latent_vars=(:z,), hyper_vars=(:θ,), 
        autodiff=MuseInference.ForwardDiffBackend()
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
    model, observed_vars=(:x,:y), latent_vars=(:z,:w), hyper_vars=(:σ,)
)
```

!!! note

    When defining Turing models to be used with MuseInference, the new-style definition 
    of Turing models is required, where the random variables do not appear as arguments 
    to the function. This is because internally, MuseInference needs
    to [`condition`](https://turinglang.github.io/DynamicPPL.jl/stable/#AbstractPPL.condition-Tuple{Model}) 
    your model on various variables.

The `autodiff` parameter should be either
`MuseInference.ForwardDiffBackend()` or
`MuseInference.ZygoteBackend()`, specifying which library to use for
automatic differenation. 

"""
function TuringMuseProblem(
    model; 
    observed_vars = (:x,),
    latent_vars = (:z,),
    hyper_vars = (:θ,),
    autodiff = AD.ForwardDiffBackend()
)

    # model is expected to be passed in conditioned on x, so grab from there
    x = ComponentVector(select(model.context.values, observed_vars))
    # vi for (z,θ) parameters in transformed space
    vi_z_θ = VarInfo(model)
    link!(vi_z_θ, SampleFromPrior())
    # model with all vars free
    model = decondition(model)
    # model for computing prior, just need any values for (x,z) to condition on here
    vars = _namedtuple(evaluate!!(model)[2])
    model_for_prior = model | select(vars, (observed_vars..., latent_vars...))
    # vi for θ in transformed space
    vi_θ = VarInfo(model_for_prior)
    link!(vi_θ, SampleFromPrior())

    TuringMuseProblem(
        autodiff,
        model,
        model_for_prior,
        vi_z_θ,
        vi_θ,
        x,
        observed_vars,
        latent_vars,
        hyper_vars
    )

end

function transform_θ(prob, θ)
    vi = deepcopy(prob.vi_θ)
    setval!(vi, θ)
    link!(vi, SampleFromPrior())
    _namedtuple(vi)
end

function inv_transform_θ(prob, θ)
    vi = deepcopy(prob.vi_θ)
    setval!(vi, θ)
    for k in keys(θ)
        settrans!(vi, true, VarName(k))
    end
    invlink!(vi, SampleFromPrior())
    _namedtuple(vi)
end

standardizeθ(prob::TuringMuseProblem, θ::NamedTuple) = ComponentVector(θ)
standardizeθ(prob::TuringMuseProblem, θ::Number) = length(prob.hyper_vars) == 1 ? ComponentVector(;θ) : error("Invalid θ type for this problem.")

function logPriorθ(prob::TuringMuseProblem, θ)
    logprior(prob.model_for_prior, VarInfo(prob.vi_θ, θ))
end

function ∇θ_logLike(prob::TuringMuseProblem, x, z, θ)
    model = condition(prob.model, x)
    first(AD.gradient(prob.autodiff, θ -> logjoint(model, VarInfo(prob.vi_z_θ, (;_namedtuple(z)..., _namedtuple(θ)...))), θ))
end

function logLike_and_∇z_logLike(prob::TuringMuseProblem, x, z, θ)
    error("Not implemented.")
end

function ẑ_at_θ(prob::TuringMuseProblem, x, z₀, θ; ∇z_logLike_atol)
    model = condition(prob.model, x)
    neglogp(z) = -logjoint(model, VarInfo(prob.vi_z_θ, (;_namedtuple(z)..., _namedtuple(θ)...)))
    soln = Optim.optimize(optim_only_fg!(neglogp, prob.autodiff), z₀, Optim.LBFGS(), Optim.Options(g_tol=∇z_logLike_atol))
    Optim.converged(soln) || warn("MAP solution failed, result could be erroneous. Try tweaking `θ₀` or `∇z_logLike_atol` argument to `muse` or fixing model.")
    soln.minimizer, soln
end

function sample_x_z(prob::TuringMuseProblem, rng::AbstractRNG, θ)
    model = condition(prob.model, inv_transform_θ(prob, θ))
    vi = VarInfo(rng, model)
    vars_constrained = map(copy, _namedtuple(vi))
    link!(vi, SampleFromPrior())
    vars_unconstrained = map(copy, _namedtuple(vi))
    (;
        x = ComponentVector(select(vars_constrained,   prob.observed_vars)),
        z = ComponentVector(select(vars_unconstrained, prob.latent_vars))
    )
end



# helped to extract parameters from a sampled model. feels like there
# should be a less hacky way to do this...
function _namedtuple(vi::VarInfo)
    map(TypedVarInfo(vi).metadata) do m
        if m.vns[1] isa VarName{<:Any,Setfield.IdentityLens} && length(m.vals)==1
            m.vals[1]
        else
            m.vals
        end
    end
end

VarInfo(vi::TypedVarInfo, x::ComponentVector) = VarInfo(vi, _namedtuple(x))

function VarInfo(vi::TypedVarInfo, x::NamedTuple)
    T = promote_type(map(eltype, values(x))..., map(eltype, _namedtuple(values(vi)))...) # if x is ForwardDiff Duals
    VarInfo(
        NamedTuple{keys(vi.metadata)}(map(keys(vi.metadata),values(vi.metadata)) do k,v
            Metadata(
                v.idcs,
                v.vns,
                v.ranges,
                atleast1d(getfield(x,k)),
                v.dists,
                v.gids,
                v.orders,
                v.flags,
            )
        end),
        Base.RefValue{T}(getlogp(vi)),
        vi.num_produce
    )
end

condition(model::Model, x::ComponentVector) = condition(model, _namedtuple(x))

atleast1d(x::Number) = [x]
atleast1d(x::AbstractVector) = x

muse!(result::MuseResult, model::Turing.Model, args...; kwargs...) = muse!(result, TuringMuseProblem(model), args...; kwargs...)
get_J!(result::MuseResult, model::Turing.Model, args...; kwargs...) = get_J!(result, TuringMuseProblem(model), args...; kwargs...)
get_H!(result::MuseResult, model::Turing.Model, args...; kwargs...) = get_H!(result, TuringMuseProblem(model), args...; kwargs...)
