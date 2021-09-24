
### Generic MPM code

abstract type AbstractMPMProblem end


## interface to be implemented by specific problem types

function ∇θ_logLike end
function logLike_and_∇z_logLike end
function sample_x_z end


# this can also be overriden by specific problems
# the default does LBFGS using the provided logLike_and_∇z_logLike
function ẑ_at_θ(prob::AbstractMPMProblem, x, θ, z₀; ∇z_logLike_atol)
    soln = optimize(Optim.only_fg(z -> .-logLike_and_∇z_logLike(prob, x, θ, z)), z₀, Optim.LBFGS(), Optim.Options(g_tol=∇z_logLike_atol))
    soln.minimizer
end



### MPM solver

function mpm(
    prob :: AbstractMPMProblem, 
    x,
    θ₀;
    rng = copy(Random.default_rng()),
    z₀ = sample_x_z(prob, copy(rng), θ₀).z,
    maxsteps = 50,
    θ_rtol = 1e-4,
    ∇z_logLike_atol = 1e-1,
    nsims = 100,
    α = 1,
    progress = false,
    pmap = _map,
    batch_size = 1,
    regularize = (θ,σθ) -> θ,
    logPrior = θ -> 0,
    H⁻¹_like = nothing,
    H⁻¹_update = :sims,
    checkpoint_filename = nothing,
    history = []
)

    θunreg = θ = θ₀
    local H⁻¹_post
    
    _rng = copy(rng)
    xz_sims = [sample_x_z(prob, _rng, θ) for i=1:nsims]
    xs = [[x];  getindex.(xz_sims, :x)]
    ẑs = [[z₀]; getindex.(xz_sims, :z)]

    # set up progress bar
    pbar = progress ? RemoteProgress(maxsteps*(nsims+1)÷batch_size, 0.1, "MPM: ") : nothing

    try
    
        for i=1:maxsteps
            
            if i>1
                _rng = copy(rng)
                xs = [[x]; [sample_x_z(prob, _rng, θ).x for i=1:nsims]]
            end

            if i>2
                Δθ = history[end].θ - history[end-1].θ
                norm(Δθ ./ θ) < θ_rtol && break
            end    

            # MPM gradient
            gẑs = pmap(xs, ẑs, fill(θ,length(xs)); batch_size) do x, ẑ_prev, θ
                ẑ = ẑ_at_θ(prob, x, θ, ẑ_prev; ∇z_logLike_atol)
                g = ∇θ_logLike(prob, x, θ, ẑ)
                progress && ProgressMeter.next!(pbar)
                (;g, ẑ)
            end
            ẑs = getindex.(gẑs, :ẑ)
            g_like_dat, g_like_sims = peel(getindex.(gẑs, :g))
            g_like = g_like_dat .- mean(g_like_sims)
            g_prior = _gradient(ForwardDiffAD(), logPrior, θ)
            g_post = g_like .+ g_prior

            # Jacobian
            if H⁻¹_like == nothing || (i>2 && H⁻¹_update == :sims)
                # if no user-provided likelihood Jacobian
                # approximation, start with a simple diagonal
                # approximation based on gradient sims
                h⁻¹_like = -1 ./ var(collect(g_like_sims))
                H⁻¹_like = h⁻¹_like isa Number ? h⁻¹_like : Diagonal(h⁻¹_like)
            elseif i > 2 && (H⁻¹_update in [:broyden, :diagonal_broyden])
                # on subsequent steps, do a Broyden's update
                Δθ = history[end].θ - history[end-1].θ
                Δg_like = history[end].g_like - history[end-1].g_like
                H⁻¹_like = H⁻¹_like + ((Δθ - H⁻¹_like * Δg_like) / (Δθ' * H⁻¹_like * Δg_like)) * Δθ' * H⁻¹_like
                if H⁻¹_update == :diagonal_broyden
                    H⁻¹_like = Diagonal(H⁻¹_like)
                end
            end

            H_prior = _hessian(ForwardDiffAD(), logPrior, θ)
            H⁻¹_post = inv(inv(H⁻¹_like) + H_prior)

            push!(history, (;θ, θunreg, g_like_dat, g_like_sims, g_like, g_prior, g_post, H⁻¹_post, H_prior, H⁻¹_like))

            # Newton-Rhapson step
            θunreg = θ .- α .* (H⁻¹_post * g_post)
            θ = regularize(θunreg, H⁻¹_post)

            (checkpoint_filename != nothing) && save(checkpoint_filename, "history", history)

        end

    finally

        progress && ProgressMeter.finish!(pbar)

    end

    θunreg, sqrt(-H⁻¹_post), history

end


function get_H(
    prob :: AbstractMPMProblem, 
    θ₀, 
    fdm :: FiniteDifferenceMethod = central_fdm(3,1); 
    ∇z_logLike_atol = 1e-8,
    rng = Random.default_rng(),
    nsims = 1, 
    step = nothing, 
    pmap = _map,
    batch_size = 1,
    pmap_over = :jac,
    progress = true,
    skip_errors = false,
    Hs = []
)

    pbar = progress ? RemoteProgress(nsims*(1+length(θ₀))÷batch_size, 0.1, "get_H: ") : nothing

    # generate simulation locally, advancing rng, and saving rng state to be reused remotely
    xs_zs_rngs = map(1:nsims) do i
        _rng = copy(rng)
        (x, z) = sample_x_z(prob, rng, θ₀)
        (x, z, _rng)
    end

    # initial fit at fiducial, used at starting points for finite difference below
    ẑ₀s_rngs = pmap(xs_zs_rngs; batch_size) do (x, z, rng)
        ẑ = ẑ_at_θ(prob, x, θ₀, z; ∇z_logLike_atol)
        progress && ProgressMeter.next!(pbar)
        (ẑ, rng)
    end

    # finite difference Jacobian
    pmap_sims, pmap_jac = (length(θ₀) > nsims) ? (_map, pmap) : (pmap, _map)
    append!(Hs, skipmissing(pmap_sims(ẑ₀s_rngs; batch_size) do (ẑ₀, rng)
        try
            return first(pjacobian(fdm, θ₀, step; pmap=pmap_jac, batch_size, pbar) do θ
                x, = sample_x_z(prob, copy(rng), θ)
                ẑ = ẑ_at_θ(prob, x, θ₀, ẑ₀; ∇z_logLike_atol)
                ∇θ_logLike(prob, x, θ₀, ẑ)
            end)
        catch err
            if skip_errors && !(err isa InterruptException)
                @warn err
                return missing
            else
                rethrow(err)
            end
        end
    end))

    mean(Hs), Hs

end


function get_J(
    prob :: AbstractMPMProblem, 
    θ₀; 
    ∇z_logLike_atol = 1e-1,
    rng = Random.default_rng(),
    nsims = 1, 
    pmap = _map,
    batch_size = 1,
    progress = true, 
    skip_errors = false,
    gs = []
)

    pbar = progress ? RemoteProgress(nsims÷batch_size, 0.1, "get_J: ") : nothing

    (xs, zs) = map(Base.vect, map(1:nsims) do i
        sample_x_z(prob, rng, θ₀)
    end...)

    append!(gs, skipmissing(pmap(xs, zs, fill(θ₀,length(xs)); batch_size) do x, z, θ₀
        try
            ẑ = ẑ_at_θ(prob, x, θ₀, z; ∇z_logLike_atol)
            g = ∇θ_logLike(prob, x, θ₀, ẑ)
            progress && ProgressMeter.next!(pbar)
            return g
        catch err
            if skip_errors && !(err isa InterruptException)
                @warn err
                return missing
            else
                rethrow(err)
            end
        end
    end))

    cov(gs), gs

end