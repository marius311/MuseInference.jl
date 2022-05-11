
### Turing interface

import .Turing
import .Turing.DynamicPPL as DynPPL
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

    TuringMuseProblem(model; params, autodiff = Turing.ADBACKEND)

Specify a MUSE problem with a [Turing](https://turing.ml) model.

The Turing model should be conditioned on the variables which comprise
the "data", and all other variables should be unconditioned. By
default, any parameter which doesn't depend on another parameter will
be estimated by MUSE, but this can be overridden by passing `params`
as a list of symbols. All other non-conditioned and non-`params`
variables will be considered the latent space.

The `autodiff` parameter should be either
`MuseInference.ForwardDiffBackend()` or
`MuseInference.ZygoteBackend()`, specifying which library to use for
automatic differenation. The default uses whatever the global
`Turing.ADBACKEND` is currently set to.


## Example

```julia
# toy hierarchical model
Turing.@model function toy()
    σ ~ Uniform(0, 1)
    θ ~ Normal(0, σ)
    z ~ MvNormal(zeros(512), exp(σ/2)*I)
    w ~ MvNormal(z, I)
    x ~ MvNormal(w, I)
    y ~ MvNormal(x, I)
    (;σ, θ, z, w, x, y)
end
sim = toy()()
model = toy() | (;sim.x, sim.y)
prob = TuringMuseProblem(model, params=(:σ, :θ))

# get solution
result = muse(prob, (σ=0.5, θ=0))
```

Here we have chosen `(σ, θ)` to be the parameters which MuseInference
will estimate (note that the default would have only chosen `σ`). The
observed data are `(x,y)` and the remaining `(z,w)` are the latent
space.

!!! note

    You can also call [`muse`](@ref), etc... directly on the model, e.g.
    `muse(model, (σ=0.5, θ=0))`, in which case the parameter names `params`
    will be read from the keys of provided the starting point.

!!! note

    The model function cannot have any of the random variables as 
    arguments, although it can have other parameters as arguments. E.g.,

    ```julia
    # not OK
    @model function toy(x)
        x ~ Normal()
        ...
    end

    # OK
    @model function toy(σ)
        x ~ Normal(σ)
        ...
    end
    ```

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
    # prevent this constructor from advancing the default RNG for more clear reproducibility
    rng = copy(Random.default_rng())
    # model is expected to be passed in conditioned on x
    x = ComponentVector(model.context.values)
    # figure out variable names
    observed = keys(x)
    latent = keys(delete(_namedtuple(DynPPL.VarInfo(rng, model)), (observed..., params...)))
    # VarInfo for (z,θ) with both transformed
    vi_z′_θ′ = DynPPL.VarInfo(rng, model)
    DynPPL.settrans!.((vi_z′_θ′,), true, _VarName.((latent..., params...)))
    # VarInfo for (z,θ) with only z transformed
    vi_z′_θ = DynPPL.VarInfo(rng, model)
    DynPPL.settrans!.((vi_z′_θ,), true, _VarName.(latent))
    # model with all vars free
    model = DynPPL.decondition(model)
    # model for computing prior, just need any values for (x,z) to condition on here
    vars = _namedtuple(DynPPL.evaluate!!(model, rng)[2])
    model_for_prior = model | select(vars, (observed..., latent...))
    # VarInfo for θ
    vi_θ = DynPPL.VarInfo(rng, model_for_prior)
    # VarInfo for transformed θ
    vi_θ′ = deepcopy(vi_θ)
    DynPPL.settrans!.((vi_θ′,), true, _VarName.(params))

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
    DynPPL.setval!(vi, θ)
    DynPPL.link!(vi, DynPPL.SampleFromPrior())
    ComponentVector(vi)
end

function inv_transform_θ(prob::TuringMuseProblem, θ)
    vi = deepcopy(prob.vi_θ)
    DynPPL.setval!(vi, θ)
    for k in keys(θ)
        DynPPL.settrans!(vi, true, _VarName(k))
    end
    DynPPL.invlink!(vi, DynPPL.SampleFromPrior())
    ComponentVector(vi)
end

standardizeθ(prob::TuringMuseProblem, θ::NamedTuple) = 1f0 * ComponentVector(θ) # ensure at least Float32
standardizeθ(prob::TuringMuseProblem, θ::Number) = 
    length(prob.hyper_vars) == 1 ? standardizeθ(prob, (;θ)) : error("Invalid θ type for this problem.")

function logPriorθ(prob::TuringMuseProblem, θ, θ_space)
    vi = is_transformed(θ_space) ? prob.vi_θ′ : prob.vi_θ
    DynPPL.logprior(prob.model_for_prior, DynPPL.VarInfo(vi, θ))
end

function ∇θ_logLike(prob::TuringMuseProblem, x, z, θ, θ_space)
    model = DynPPL.condition(prob.model, x)
    vi = is_transformed(θ_space) ? prob.vi_z′_θ′ : prob.vi_z′_θ
    first(AD.gradient(prob.autodiff, θ -> DynPPL.logjoint(model, DynPPL.VarInfo(vi, z, θ)), θ))
end

function ẑ_at_θ(prob::TuringMuseProblem, x, z₀, θ; ∇z_logLike_atol)
    model = DynPPL.condition(prob.model, x)
    neglogp(z) = -DynPPL.logjoint(model, DynPPL.VarInfo(prob.vi_z′_θ, z, θ))
    soln = Optim.optimize(optim_only_fg!(neglogp, prob.autodiff), z₀, Optim.LBFGS(), Optim.Options(g_tol=∇z_logLike_atol))
    _check_optim_soln(soln)
    soln.minimizer, soln
end

function sample_x_z(prob::TuringMuseProblem, rng::AbstractRNG, θ)
    model = DynPPL.condition(prob.model, θ)
    vi = DynPPL.VarInfo(rng, model)
    vars_untransformed = map(copy, _namedtuple(vi))
    DynPPL.link!(vi, DynPPL.SampleFromPrior())
    vars_transformed = map(copy, _namedtuple(vi))
    (;
        x = ComponentVector(select(vars_untransformed, prob.observed_vars)),
        z = ComponentVector(select(vars_transformed,   prob.latent_vars))
    )
end



# helped to extract parameters from a sampled model. feels like there
# should be a less hacky way to do this...
function _namedtuple(vi::DynPPL.VarInfo)
    map(DynPPL.TypedVarInfo(vi).metadata) do m
        if m.vns[1] isa DynPPL.VarName{<:Any,Setfield.IdentityLens} && length(m.vals)==1
            m.vals[1]
        else
            m.vals
        end
    end
end

ComponentVector(vi::DynPPL.VarInfo) = ComponentVector(_namedtuple(vi))

function DynPPL.VarInfo(vi::DynPPL.TypedVarInfo, x::Union{NamedTuple,ComponentVector}, xs::Union{NamedTuple,ComponentVector}...)
    DynPPL.VarInfo(vi, merge(map(_namedtuple, (x, xs...))...))
end

function DynPPL.VarInfo(vi::DynPPL.TypedVarInfo, x::NamedTuple)
    T = promote_type(map(eltype, values(x))..., map(eltype, _namedtuple(values(vi)))...) # if x is ForwardDiff Duals
    DynPPL.VarInfo(
        NamedTuple{keys(vi.metadata)}(map(keys(vi.metadata),values(vi.metadata)) do k,v
            DynPPL.Metadata(
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
        Base.RefValue{T}(DynPPL.getlogp(vi)),
        vi.num_produce
    )
end

DynPPL.condition(model::DynPPL.Model, x::ComponentVector) = DynPPL.condition(model, _namedtuple(x))
_VarName(x::Symbol) = DynPPL.VarName{x}()

atleast1d(x::Number) = [x]
atleast1d(x::AbstractVector) = x

_params_from_θ₀(θ₀::Number) = (:θ,)
_params_from_θ₀(θ₀) = keys(θ₀)

function muse!(result::MuseResult, model::DynPPL.Model, θ₀ = result.θ; kwargs...)
    muse!(result, TuringMuseProblem(model, params=_params_from_θ₀(θ₀)), θ₀; kwargs...)
end
function get_J!(result::MuseResult, model::DynPPL.Model, θ₀ = result.θ; kwargs...)
    get_J!(result, TuringMuseProblem(model, params=_params_from_θ₀(θ₀)), θ₀; kwargs...)
end
function get_H!(result::MuseResult, model::DynPPL.Model, θ₀ = result.θ; kwargs...)
    get_H!(result, TuringMuseProblem(model, params=_params_from_θ₀(θ₀)), θ₀; kwargs...)
end
