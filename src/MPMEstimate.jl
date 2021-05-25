module MPMEstimate

using Distributions
using ForwardDiff
using Optim
using ProgressMeter
using Random
using Requires
using Setfield

export MPMProblem, mpm


### Generic MPM code

abstract type AbstractMPMProblem end


## interface to be implemented by specific problem types

function ∇θ_logP(prob::AbstractMPMProblem, x, θ, z) end
function logP_and_∇z_logP(prob::AbstractMPMProblem, x, θ, z) end
function sample_x_z(prob::AbstractMPMProblem, rng::AbstractRNG, θ) end


## generic AbstractMPMProblem solution

function ẑ_at_θ(prob::AbstractMPMProblem, x, θ, z₀)
    soln = optimize(Optim.only_fg(z -> .-logP_and_∇z_logP(prob, x, θ, z)), z₀, Optim.LBFGS(), Optim.Options(f_tol=1e-3))
    soln.minimizer
end

function mpm(
    prob :: AbstractMPMProblem, 
    x,
    θ₀;
    rng = Random.default_rng(),
    z₀ = sample_x_z(prob, copy(rng), θ₀).z,
    nsteps = 3,
    nsims = 100,
    α = 0.5
)

    θ = θ₀
    local g_sims₀, σθ
    history = Any[(;θ)]
    rng = copy(rng)

    @showprogress for i=1:nsteps

        ẑ = ẑ_at_θ(prob, x, θ, z₀)
        g_dat = ∇θ_logP(prob, x, θ, ẑ)

        _rng = copy(rng)
        g_sims = map(1:nsims) do i
            x_sim, z_sim = sample_x_z(prob, _rng, θ)
            ∇θ_logP(prob, x_sim, θ, ẑ_at_θ(prob, x_sim, θ, z_sim))
        end
        if i==1
            g_sims₀ = g_sims
        end

        σθ = 1 / std(g_sims₀)
        θ = θ + α * (var(g_sims₀) \ (g_dat - mean(g_sims)))

        z₀ = ẑ
        push!(history, (;θ, g_dat, g_sims, σθ))

    end

    θ, σθ, history

end

### Autodiff types

abstract type ADBackend end
struct ForwardDiffAD <: ADBackend end
struct ZygoteAD <: ADBackend end

_gradient(autodiff, f, x) = _val_and_gradient(autodiff, f, x)[2]
_val_and_gradient(autodiff, f, x) = error("Run `using $(string(autodiff)[1:end-4])` to enable $autodiff")

# ForwardDiff (loaded by default)
_gradient(::ForwardDiffAD, f, x::Real) = ForwardDiff.derivative(f, x)
_gradient(::ForwardDiffAD, f, x) = ForwardDiff.gradient(f, x)
_val_and_gradient(::ForwardDiffAD, f, x) = f(x), _gradient(ForwardDiffAD(), f, x)

# Zygote
@init @require Zygote="e88e6eb3-aa80-5325-afca-941959d7151f" begin
    function _val_and_gradient(::ZygoteAD, f, x)
        y, back = pullback(f, x)
        y, back(sensitivity(y))
    end
end



### Manual MPMProblem

struct MPMProblem{S,Gθ,GZ} <: AbstractMPMProblem
    sample_x_z :: S
    ∇θ_logP :: Gθ
    logP_and_∇z_logP :: GZ
end

function MPMProblem(sample_x_z, logP, autodiff::ADBackend=ForwardDiffAD())
    MPMProblem(
        sample_x_z,
        (x,θ,z) -> _gradient(autodiff, θ -> logP(x,θ,z), θ),
        (x,θ,z) -> _val_and_gradient(autodiff, z -> logP(x,θ,z), z)
    )
end

∇θ_logP(prob::MPMProblem, x, θ, z) = prob.∇θ_logP(x, θ, z)
logP_and_∇z_logP(prob::MPMProblem, x, θ, z) = prob.logP_and_∇z_logP(x, θ, z)
sample_x_z(prob::MPMProblem, rng::AbstractRNG, θ) = prob.sample_x_z(rng, θ)


### TuringMPMProblem

@init @require Turing="fce5fe82-541a-59a6-adf8-730c64b5f9a0" begin

    import .Turing
    using .Turing: VarInfo, gradient_logp, SampleFromPrior, DefaultContext
    
    export TuringMPMProblem

    struct TuringMPMProblem{M,P} <: AbstractMPMProblem
        model :: M
        model_for_sampling_prior :: P
    end

    TuringMPMProblem(model) = TuringMPMProblem(model, @set(model.args = _args_for_sampling_prior(model.args)))

    _args_for_sampling_prior(x::NamedTuple) = map(_args_for_sampling_prior, x)
    _args_for_sampling_prior(x::Real) = missing
    _args_for_sampling_prior(x::AbstractVector{T}) where {T} = map!(_args_for_sampling_prior, similar(x,Union{T,Missing}), x)
    _args_for_sampling_prior(x) = x


    # todo: figure out how to use Selector (?) to generically get the
    # right parameter sub-spaces rather than manually indexing into these
    # vectors

    function ∇θ_logP(prob::TuringMPMProblem, x, θ, z)
        model = prob.model
        @set! model.args.x = x
        _, g = gradient_logp(Turing.ForwardDiffAD{1}(), [θ; z], VarInfo(model), model)
        θ isa Real ? g[1] : g[1:length(θ)]
    end

    function logP_and_∇z_logP(prob::TuringMPMProblem, x, θ, z)
        model = prob.model
        @set! model.args.x = x
        f, g = gradient_logp(Turing.ForwardDiffAD{1}(), [θ; z], VarInfo(model), model)
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

end



end