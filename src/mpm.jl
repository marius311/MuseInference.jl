
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
    regularize = (θ,σθ) -> θ,
)

    θ = θ₀
    local σθ, θunreg
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
            θunreg = @. θ + α * σθ^2 * (g_dat - $mean(g_sims))
            θ = regularize(θunreg, σθ)

            push!(history, (;θ, θunreg, g_dat, g_sims, σθ))

        end

    finally

        progress && put!(update_pbar, false)

    end

    θunreg, σθ, history

end


function get_H(prob::AbstractMPMProblem, θ₀; nsims=1, ϵ=0.05, map=map)
    
    fdm = central_fdm(2, 1, adapt=0)

    # do initial fit at θ₀ for each sim, used as starting points below
    ẑ₀s_rngs = map(1:nsims) do i
        _rng = copy(rng)
        x, z = sample_x_z(prob, rng, θ₀)
        ẑ = ẑ_at_θ(prob, x, θ₀, z)
        (ẑ, _rng)
    end

    mean(ẑ₀s_rngs) do ẑ₀, rng
        pjacobian(fdm, θ₀, ϵ) do θ
            x, = sample_x_z(prob, copy(rng), θ)
            ẑ = ẑ_at_θ(prob, x, θ₀, ẑ₀)
            ∇θ_logP(prob, x, θ₀, ẑ)
        end
    end

end