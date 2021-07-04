
### Generic MPM code

abstract type AbstractMPMProblem end


## interface to be implemented by specific problem types

function ∇θ_logLike(prob::AbstractMPMProblem, x, θ, z) end
function logLike_and_∇z_logLike(prob::AbstractMPMProblem, x, θ, z) end
function sample_x_z(prob::AbstractMPMProblem, rng::AbstractRNG, θ) end


## generic AbstractMPMProblem solution

function ẑ_at_θ(prob::AbstractMPMProblem, x, θ, z₀)
    soln = optimize(Optim.only_fg(z -> .-logLike_and_∇z_logLike(prob, x, θ, z)), z₀, Optim.LBFGS())
    soln.minimizer
end

function mpm(
    prob :: AbstractMPMProblem, 
    x,
    θ₀;
    rng = copy(Random.default_rng()),
    z₀ = sample_x_z(prob, copy(rng), θ₀).z,
    maxsteps = 50,
    θ_rtol = 1e-4,
    nsims = 100,
    α = 1,
    progress = false,
    map = map,
    regularize = (θ,σθ) -> θ,
    logPrior = θ -> 0,
    H⁻¹_like = nothing
)

    θunreg = θ = θ₀
    local H⁻¹_post
    history = []
    
    _rng = copy(rng)
    xz_sims = [sample_x_z(prob, _rng, θ) for i=1:nsims]
    xs = [[x];  getindex.(xz_sims, :x)]
    ẑs = [[z₀]; getindex.(xz_sims, :z)]

    # set up progress bar
    if progress
        pbar = Progress(maxsteps*(nsims+1), 0.1, "MPM: ")
        ProgressMeter.update!(pbar)
        update_pbar = RemoteChannel(()->Channel{Bool}(), 1)
        @async while take!(update_pbar)
            next!(pbar)
        end
    end

    try
    
        for i=1:maxsteps
            
            if i>1
                _rng = copy(rng)
                xs = [[x]; [sample_x_z(prob, _rng, θ).x for i=1:nsims]]
            end

            # MPM gradient
            gẑs = map(xs, ẑs) do x, ẑ_prev
                ẑ = ẑ_at_θ(prob, x, θ, ẑ_prev)
                g = ∇θ_logLike(prob, x, θ, ẑ)
                progress && put!(update_pbar, true)
                (;g, ẑ)
            end
            ẑs = getindex.(gẑs, :ẑ)
            g_like_dat, g_like_sims = peel(getindex.(gẑs, :g))
            g_like = g_like_dat .- mean(g_like_sims)
            g_prior = _gradient(ForwardDiffAD(), logPrior, θ)
            g_post = g_like .+ g_prior

            # Jacobian
            if H⁻¹_like == nothing
                # if no user-provided likelihood Jacobian
                # approximation, start with a simple diagonal
                # approximation based on gradient sims
                h⁻¹_like = -1 ./ var(collect(g_like_sims))
                H⁻¹_like = h⁻¹_like isa Number ? h⁻¹_like : Diagonal(h⁻¹_like)
            elseif i>2
                # on subsequent steps, do a Broyden's update
                Δθ = history[end].θ - history[end-1].θ
                norm(Δθ ./ θ) < θ_rtol && break
                Δg_like = history[end].g_like - history[end-1].g_like
                H⁻¹_like = H⁻¹_like + ((Δθ - H⁻¹_like * Δg_like) / (Δθ' * H⁻¹_like * Δg_like)) * Δθ' * H⁻¹_like
            end

            H_prior = _hessian(ForwardDiffAD(), logPrior, θ)
            H⁻¹_post = inv(inv(H⁻¹_like) + H_prior)

            push!(history, (;θ, θunreg, g_like_dat, g_like_sims, g_like, g_prior, g_post, H⁻¹_post, H_prior, H⁻¹_like))

            # Newton-Rhapson step
            θunreg = θ .- α .* (H⁻¹_post * g_post)
            θ = regularize(θunreg, H⁻¹_post)

        end

    finally

        progress && put!(update_pbar, false)

    end

    θunreg, sqrt(-H⁻¹_post), history

end


function get_H(
    prob :: AbstractMPMProblem, 
    θ₀, 
    fdm :: FiniteDifferenceMethod = central_fdm(3,1); 
    rng = copy(Random.default_rng()),
    nsims = 1, 
    step = nothing, 
    pmap = map,
    progress = true
)

    if progress
        pbar = Progress(nsims*(1+length(θ₀)), 0.1, "get_H: ")
        ProgressMeter.update!(pbar)
        pbar_ch = RemoteChannel(()->Channel{Bool}(), 1)
        @async while take!(pbar_ch)
            next!(pbar)
        end
        update_pbar = () -> put!(pbar_ch, true)
    else
        update_pbar = () -> nothing
    end

    # do initial fit at θ₀ for each sim, used as starting points below
    ẑ₀s_rngs = pmap(1:nsims) do i
        _rng = copy(rng)
        x, z = sample_x_z(prob, rng, θ₀)
        ẑ = ẑ_at_θ(prob, x, θ₀, z)
        update_pbar()
        (ẑ, _rng)
    end

    mean(map(ẑ₀s_rngs) do (ẑ₀, rng)
        first(pjacobian(fdm, θ₀, step; pmap, update_pbar) do θ
            x, = sample_x_z(prob, copy(rng), θ)
            ẑ = ẑ_at_θ(prob, x, θ₀, ẑ₀)
            ∇θ_logLike(prob, x, θ₀, ẑ)
        end)
    end)

end