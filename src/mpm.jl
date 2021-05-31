
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
    α = 0.7,
    progress = false,
    map = map,
)

    θ = θ₀
    local σθ
    history = Any[(;θ)]
    
    _rng = copy(rng)
    xz_sims = [sample_x_z(prob, _rng, θ) for i=1:nsims]
    xs = [[x];  getindex.(xz_sims, :x)]
    ẑs = [[z₀]; getindex.(xz_sims, :z)]

    # set up progress bar
    if progress
        pbar = Progress(nsteps*(nsims+1), 0.1, "MPM: ")
        ProgressMeter.update!(pbar)
        update_pbar = RemoteChannel(()->Channel{Bool}(), 1)
        @async while take!(update_pbar)
            next!(pbar)
        end
    end

    try
    
        for i=1:nsteps
            
            if i>1
                _rng = copy(rng)
                xs = [[x]; [sample_x_z(prob, _rng, θ).x for i=1:nsims]]
            end

            gẑs = map(xs, ẑs) do x, ẑ_prev
                ẑ = ẑ_at_θ(prob, x, θ, ẑ_prev)
                g = ∇θ_logP(prob, x, θ, ẑ)
                progress && put!(update_pbar, true)
                (;g, ẑ)
            end

            ẑs = getindex.(gẑs, :ẑ)
            g_dat, g_sims = peel(getindex.(gẑs, :g))

            σθ = 1 ./ std(collect(g_sims))
            θ = @. θ + α * σθ^2 * (g_dat - $mean(g_sims))

            push!(history, (;θ, g_dat, g_sims, σθ))

        end

    finally

        progress && put!(update_pbar, false)

    end

    θ, σθ, history

end
