
### Generic MPM code

abstract type AbstractMPMProblem end


## interface to be implemented by specific problem types

function ∇θ_logP(prob::AbstractMPMProblem, x, θ, z) end
function logP_and_∇z_logP(prob::AbstractMPMProblem, x, θ, z) end
function sample_x_z(prob::AbstractMPMProblem, rng::AbstractRNG, θ) end


## generic AbstractMPMProblem solution

function ẑ_at_θ(prob::AbstractMPMProblem, x, θ, z₀)
    soln = optimize(Optim.only_fg(z -> .-logP_and_∇z_logP(prob, x, θ, z)), z₀, Optim.LBFGS())
    soln.minimizer
end

function mpm(
    prob :: AbstractMPMProblem, 
    x,
    θ₀;
    rng = Random.default_rng(),
    z₀ = sample_x_z(prob, copy(rng), θ₀).z,
    nsteps = 5,
    nsims = 100,
    α = 0.7
)

    θ = θ₀
    local g_sims₀, σθ
    history = Any[(;θ)]
    rng = copy(rng)

    for i=1:nsteps

        ẑ = ẑ_at_θ(prob, x, θ, z₀)
        g_dat = ∇θ_logP(prob, x, θ, ẑ)

        _rng = copy(rng)
        g_sims = map(1:nsims) do i
            x_sim, z_sim = sample_x_z(prob, _rng, θ)
            ∇θ_logP(prob, x_sim, θ, ẑ_at_θ(prob, x_sim, θ, z_sim))
        end

        σθ = 1 / std(g_sims)
        θ = θ + α * (var(g_sims) \ (g_dat - mean(g_sims)))

        z₀ = ẑ
        push!(history, (;θ, g_dat, g_sims, σθ))

    end

    θ, σθ, history

end
