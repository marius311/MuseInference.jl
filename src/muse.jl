
### Generic MUSE code

abstract type AbstractMuseProblem end


## interface to be implemented by specific problem types

function ∇θ_logLike end
function logLike_and_∇z_logLike end
function sample_x_z end
logPriorθ(prob::AbstractMuseProblem, θ) = 0


# this can also be overriden by specific problems
# the default does LBFGS using the provided logLike_and_∇z_logLike
function ẑ_at_θ(prob::AbstractMuseProblem, x, θ, z₀; ∇z_logLike_atol)
    soln = optimize(Optim.only_fg(z -> .-logLike_and_∇z_logLike(prob, x, θ, z)), z₀, Optim.LBFGS(), Optim.Options(g_tol=∇z_logLike_atol))
    soln.minimizer, soln
end


### MUSE result

Base.@kwdef mutable struct MuseResult
    θ = nothing
    σθ = nothing
    F = nothing
    history = []
    H = nothing
    J = nothing
    gs = []
    Hs = []
    rng = nothing
end


### MUSE solver

muse(args...; kwargs...) = muse!(MuseResult(), args...; kwargs...)

function muse!(
    result :: MuseResult,
    prob :: AbstractMuseProblem, 
    θ₀ = nothing;
    rng = nothing,
    z₀ = nothing,
    maxsteps = 50,
    θ_rtol = 1e-1,
    ∇z_logLike_atol = 1e-2,
    nsims = 100,
    α = 0.7,
    progress = false,
    pmap = _map,
    batch_size = 1,
    regularize = identity,
    logPrior = θ -> 0,
    H⁻¹_like = nothing,
    H⁻¹_update = :sims,
    broyden_memory = Inf,
    checkpoint_filename = nothing,
)

    rng = @something(rng, result.rng, copy(Random.default_rng()))
    θunreg = θ = θ₀ = @something(result.θ, θ₀)
    z₀ = @something(z₀, sample_x_z(prob, copy(rng), θ₀).z)
    local H⁻¹_post
    history = result.history
    
    result.rng = _rng = copy(rng)
    xz_sims = [sample_x_z(prob, _rng, θ) for i=1:nsims]
    xs = [[prob.x];  getindex.(xz_sims, :x)]
    ẑs = [[z₀];      getindex.(xz_sims, :z)]

    # set up progress bar
    pbar = progress ? RemoteProgress((maxsteps-length(result.history))*(nsims+1)÷batch_size, 0.1, "MUSE: ") : nothing

    try
    
        for i = (length(result.history)+1):maxsteps
            
            t₀ = now()

            if i > 1
                _rng = copy(rng)
                xs = [[prob.x]; [sample_x_z(prob, _rng, θ).x for i=1:nsims]]
            end

            if i > 2
                Δθ = history[end].θ - history[end-1].θ
                norm(Δθ ./ θ) < θ_rtol && break
            end

            # MUSE gradient
            gẑs = pmap(xs, ẑs, fill(θ,length(xs)); batch_size) do x, ẑ_prev, θ
                local ẑ, history = ẑ_at_θ(prob, x, θ, ẑ_prev; ∇z_logLike_atol)
                g = ∇θ_logLike(prob, x, θ, ẑ)
                progress && ProgressMeter.next!(pbar)
                (;g, ẑ, history)
            end
            ẑs = getindex.(gẑs, :ẑ)
            ẑ_history_dat, ẑ_history_sims = peel(getindex.(gẑs, :history))
            g_like_dat, g_like_sims = peel(getindex.(gẑs, :g))
            g_like = g_like_dat .- mean(g_like_sims)
            g_prior = AD.gradient(AD.ForwardDiffBackend(), θ -> logPriorθ(prob, θ), θ)[1]
            g_post = g_like .+ g_prior

            # Jacobian
            h⁻¹_like_sims = -1 ./ var(collect(g_like_sims))
            H⁻¹_like_sims = h⁻¹_like_sims isa Number ? h⁻¹_like_sims : Diagonal(h⁻¹_like_sims)
            if (H⁻¹_like == nothing) || (H⁻¹_update == :sims)
                H⁻¹_like = H⁻¹_like_sims
            elseif i > 2 && (H⁻¹_update in [:broyden, :diagonal_broyden])
                # on subsequent steps, do a Broyden's update using at
                # most the previous `broyden_memory` steps
                j₀ = Int(max(2, i - broyden_memory))
                H⁻¹_like = history[j₀-1].H⁻¹_like_sims
                for j = j₀:i-1
                    Δθ      = history[j].θ      - history[j-1].θ
                    Δg_like = history[j].g_like - history[j-1].g_like
                    H⁻¹_like = H⁻¹_like + ((Δθ - H⁻¹_like * Δg_like) / (Δθ' * H⁻¹_like * Δg_like)) * Δθ' * H⁻¹_like
                    if H⁻¹_update == :diagonal_broyden
                        H⁻¹_like = Diagonal(H⁻¹_like)
                    end
                end
            end

            H_prior = AD.hessian(AD.ForwardDiffBackend(), θ -> logPriorθ(prob, θ), θ)[1]
            H⁻¹_post = inv(inv(H⁻¹_like) + H_prior)

            t = now() - t₀
            push!(
                history, 
                (;θ, θunreg, 
                g_like_dat, g_like_sims, g_like, g_prior, g_post, 
                H⁻¹_post, H_prior, H⁻¹_like, H⁻¹_like_sims, 
                ẑ_history_dat, ẑ_history_sims, t)
            )

            # Newton-Rhapson step
            θunreg = θ .- α .* (H⁻¹_post * g_post)
            θ = regularize(θunreg)

            (checkpoint_filename != nothing) && save(checkpoint_filename, "result", result)

        end

    finally

        progress && ProgressMeter.finish!(pbar)

    end

    result.θ = θunreg
    result

end



function get_H!(
    result :: MuseResult,
    prob :: AbstractMuseProblem, 
    θ₀ = result.θ, 
    fdm :: FiniteDifferenceMethod = central_fdm(3,1); 
    ∇z_logLike_atol = 1e-8,
    rng = Random.default_rng(),
    nsims = 10, 
    step = nothing, 
    pmap = _map,
    batch_size = 1,
    pmap_over = :auto,
    progress = false,
    skip_errors = false,
)

    nsims_remaining = nsims - length(result.Hs)
    (nsims_remaining <= 0) && return
    pbar = progress ? RemoteProgress(nsims_remaining*(1+length(θ₀))÷batch_size, 0.1, "get_H: ") : nothing

    # generate simulation locally, advancing rng, and saving rng state to be reused remotely
    xs_zs_rngs = map(1:nsims_remaining) do i
        _rng = copy(rng)
        (x, z) = sample_x_z(prob, rng, θ₀)
        (x, z, _rng)
    end

    # initial fit at fiducial, used at starting points for finite difference below
    ẑ₀s_rngs = pmap(xs_zs_rngs; batch_size) do (x, z, rng)
        ẑ, = ẑ_at_θ(prob, x, θ₀, z; ∇z_logLike_atol)
        progress && ProgressMeter.next!(pbar)
        (ẑ, rng)
    end

    # finite difference Jacobian
    pmap_sims, pmap_jac = (pmap_over == :jac || (pmap_over == :auto && length(θ₀) > nsims_remaining)) ? (_map, pmap) : (pmap, _map)
    append!(result.Hs, skipmissing(pmap_sims(ẑ₀s_rngs; batch_size) do (ẑ₀, rng)
        try
            return first(pjacobian(fdm, θ₀, step; pmap=pmap_jac, batch_size, pbar) do θ
                x, = sample_x_z(prob, copy(rng), θ)
                ẑ, = ẑ_at_θ(prob, x, θ₀, ẑ₀; ∇z_logLike_atol)
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
 
    result.H = (θ₀ isa Number) ? mean(first.(result.Hs)) :  mean(result.Hs)
    finalize_result!(result, prob)

end


function get_J!(
    result :: MuseResult,
    prob :: AbstractMuseProblem, 
    θ₀ = result.θ; 
    ∇z_logLike_atol = 1e-1,
    rng = Random.default_rng(),
    nsims = 100, 
    pmap = _map,
    batch_size = 1,
    progress = false, 
    skip_errors = false,
    covariance_method = LinearShrinkage(target=DiagonalCommonVariance(), shrinkage=:rblw),
)

    nsims_remaining = nsims - length(result.gs)
    (nsims_remaining <= 0) && return
    pbar = progress ? RemoteProgress(nsims_remaining÷batch_size, 0.1, "get_J: ") : nothing

    (xs, zs) = map(Base.vect, map(1:nsims_remaining) do i
        sample_x_z(prob, rng, θ₀)
    end...)

    append!(result.gs, skipmissing(pmap(xs, zs, fill(θ₀,length(xs)); batch_size) do x, z, θ₀
        try
            ẑ, = ẑ_at_θ(prob, x, θ₀, z; ∇z_logLike_atol)
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

    result.J = (θ₀ isa Number) ? var(result.gs) : cov(covariance_method, identity.(result.gs))
    finalize_result!(result, prob)

end


function finalize_result!(result::MuseResult, prob::AbstractMuseProblem)
    @unpack H, J = result
    if H != nothing && J != nothing
        H_prior = -AD.hessian(AD.ForwardDiffBackend(), θ -> logPriorθ(prob, θ), result.θ)[1]
        result.F = F = H' * inv(J) * H + H_prior
        if F isa Number
            result.σθ = sqrt.(1 ./ F)
        else
            result.σθ = sqrt.(diag(inv(F)))
        end
    end
    result
end
