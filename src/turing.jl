
### Turing interface

import .Turing
using .Turing: VarInfo, TypedVarInfo, tonamedtuple, decondition, logprior, 
    logjoint, VarName, SampleFromPrior, link!, invlink!, MAP, OptimLogDensity, 
    DefaultContext, SimpleVarInfo, LikelihoodContext, condition
using .Turing.ModeEstimation: transform!
using .Turing.DynamicPPL: evaluate!!, setval!, settrans!, Model, Metadata, getlogp

import .Turing.DynamicPPL: VarInfo, condition

import ComponentArrays: ComponentVector

export TuringMuseProblem


struct TuringMuseProblem{A<:AD.AbstractBackend, M<:Turing.Model} <: AbstractMuseProblem
    
    autodiff :: A
    model :: M
    model_for_prior
    vi_z′_θ
    vi_z′_θ′
    vi_θ
    vi_θ′
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
    params = (:θ,),
    autodiff = nothing,
)

    # set backend based on Turing's by default
    if autodiff == nothing
        if Turing.ADBACKEND[] == :zygote
            autodiff = ZygoteBackend()
        elseif Turing.ADBACKEND[] == :forwarddiff
            autodiff = ForwardDiffBackend()
        else
            error("Unsupposed backend from Turing: $(Turing.ADBACKEND)")
        end
    end
    # model is expected to be passed in conditioned on x
    x = ComponentVector(model.context.values)
    # figure out variable names
    observed = keys(x)
    latent = keys(delete(_namedtuple(VarInfo(model)), (observed..., params...)))
    # VarInfo for (z,θ) with both transformed
    vi_z′_θ′ = VarInfo(model)
    settrans!.((vi_z′_θ′,), true, VarName.((latent..., params...)))
    # VarInfo for (z,θ) with only z transformed
    vi_z′_θ = VarInfo(model)
    settrans!.((vi_z′_θ,), true, VarName.(latent))
    # model with all vars free
    model = decondition(model)
    # model for computing prior, just need any values for (x,z) to condition on here
    vars = _namedtuple(evaluate!!(model)[2])
    model_for_prior = model | select(vars, (observed..., latent...))
    # VarInfo for θ
    vi_θ = VarInfo(model_for_prior)
    # VarInfo for transformed θ
    vi_θ′ = deepcopy(vi_θ)
    settrans!.((vi_θ′,), true, VarName.(params))

    TuringMuseProblem(
        autodiff,
        model,
        model_for_prior,
        vi_z′_θ,
        vi_z′_θ′,
        vi_θ,
        vi_θ′,
        x,
        observed,
        latent,
        params
    )

end

function transform_θ(prob::TuringMuseProblem, θ)
    vi = deepcopy(prob.vi_θ)
    setval!(vi, θ)
    link!(vi, SampleFromPrior())
    ComponentVector(vi)
end

function inv_transform_θ(prob::TuringMuseProblem, θ)
    vi = deepcopy(prob.vi_θ)
    setval!(vi, θ)
    for k in keys(θ)
        settrans!(vi, true, VarName(k))
    end
    invlink!(vi, SampleFromPrior())
    ComponentVector(vi)
end

standardizeθ(prob::TuringMuseProblem, θ::NamedTuple) = ComponentVector(θ)
standardizeθ(prob::TuringMuseProblem, θ::Number) = length(prob.hyper_vars) == 1 ? ComponentVector(;θ) : error("Invalid θ type for this problem.")

function logPriorθ(prob::TuringMuseProblem, θ, θ_space)
    vi = is_transformed(θ_space) ? prob.vi_θ′ : prob.vi_θ
    logprior(prob.model_for_prior, VarInfo(vi, θ))
end

function ∇θ_logLike(prob::TuringMuseProblem, x, z, θ, θ_space)
    model = condition(prob.model, x)
    vi = is_transformed(θ_space) ? prob.vi_z′_θ′ : prob.vi_z′_θ
    first(AD.gradient(prob.autodiff, θ -> logjoint(model, VarInfo(vi, z, θ)), θ))
end

function ẑ_at_θ(prob::TuringMuseProblem, x, z₀, θ; ∇z_logLike_atol)
    model = condition(prob.model, x)
    neglogp(z) = -logjoint(model, VarInfo(prob.vi_z′_θ, z, θ))
    soln = Optim.optimize(optim_only_fg!(neglogp, prob.autodiff), z₀, Optim.LBFGS(), Optim.Options(g_tol=∇z_logLike_atol))
    _check_optim_soln(soln)
    soln.minimizer, soln
end

function sample_x_z(prob::TuringMuseProblem, rng::AbstractRNG, θ)
    model = condition(prob.model, θ)
    vi = VarInfo(rng, model)
    vars_untransformed = map(copy, _namedtuple(vi))
    link!(vi, SampleFromPrior())
    vars_transformed = map(copy, _namedtuple(vi))
    (;
        x = ComponentVector(select(vars_untransformed, prob.observed_vars)),
        z = ComponentVector(select(vars_transformed,   prob.latent_vars))
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

ComponentVector(vi::VarInfo) = ComponentVector(_namedtuple(vi))

function VarInfo(vi::TypedVarInfo, x::Union{NamedTuple,ComponentVector}, xs::Union{NamedTuple,ComponentVector}...)
    VarInfo(vi, merge(map(_namedtuple, (x, xs...))...))
end

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

function muse!(result::MuseResult, model::Turing.Model, θ₀ = result.θ; kwargs...)
    muse!(result, TuringMuseProblem(model, params=keys(θ₀)), θ₀; kwargs...)
end
function get_J!(result::MuseResult, model::Turing.Model, θ₀ = result.θ; kwargs...)
    get_J!(result, TuringMuseProblem(model, params=keys(θ₀)), θ₀; kwargs...)
end
function get_H!(result::MuseResult, model::Turing.Model, θ₀ = result.θ; kwargs...)
    get_H!(result, TuringMuseProblem(model, params=keys(θ₀)), θ₀; kwargs...)
end
