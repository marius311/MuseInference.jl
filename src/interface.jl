
## interface to be implemented by specific problem types

abstract type AbstractMuseProblem end

struct Transformedθ end
struct UnTransformedθ end
is_transformed(::Transformedθ) = true
is_transformed(::UnTransformedθ) = false

"""
"""
function ∇θ_logLike end
# ∇θ_logLike(prob::AbstractMuseProblem, x, z, θ, ::Transformedθ) = ∇θ_logLike(prob, x, z, inv_transform_θ(prob, θ), UnTransformedθ())

function logLike_and_∇z_logLike end
function sample_x_z end

transform_θ(prob::AbstractMuseProblem, θ) = θ
inv_transform_θ(prob::AbstractMuseProblem, θ) = θ

logPriorθ(prob::AbstractMuseProblem, θ, θ_space) = 0
# logPriorθ(prob::AbstractMuseProblem, θ, ::Transformedθ) = logPriorθ(prob::AbstractMuseProblem, inv_transform_θ(prob, θ), UnTransformedθ())

standardizeθ(prob::AbstractMuseProblem, θ) = θ

# this can also be overriden by specific problems
# the default does LBFGS using the provided logLike_and_∇z_logLike
function ẑ_at_θ(prob::AbstractMuseProblem, x, z₀, θ; ∇z_logLike_atol)
    soln = optimize(Optim.only_fg(z -> .-logLike_and_∇z_logLike(prob, x, z, θ)), z₀, Optim.LBFGS(), Optim.Options(g_tol=∇z_logLike_atol))
    _check_optim_soln(soln)
    soln.minimizer, soln
end

function _check_optim_soln(soln)
    Optim.converged(soln) || warn("MAP solution failed, result could be erroneous. Try tweaking `θ₀` or `∇z_logLike_atol` arguments to `muse` or fixing model.")
    isfinite(soln.minimum) || error("MAP solution failed with logjoint(MAP)=$(soln.minimum).")
end


function check_self_consistency(
    prob, 
    θ;
    fdm = central_fdm(3, 1),
    atol = 1e-3,
    rng = Random.default_rng()
)

    θ = standardizeθ(prob, θ)
    x, z = sample_x_z(prob, rng, θ)
    # volume factor which is added by transformations. dont assume the
    # transformation is AD-able (eg it isnt for Turing)
    J(θ) = FiniteDifferences.jacobian(fdm, θ -> transform_θ(prob, θ), θ)[1]
    V(θ) = logdet(J(θ))
    ∇θ_V(θ) = FiniteDifferences.grad(fdm, V, θ)[1]
    @testset "Self-consistency" begin
        @test inv_transform_θ(prob, transform_θ(prob, θ)) ≈ θ  atol=atol
        @test logPriorθ(prob, θ, UnTransformedθ()) ≈ logPriorθ(prob, transform_θ(prob, θ), Transformedθ()) + V(θ)  atol=atol
        @test ∇θ_logLike(prob, x, z, θ, UnTransformedθ()) ≈ J(θ)' * ∇θ_logLike(prob, x, z, transform_θ(prob, θ), Transformedθ()) + ∇θ_V(θ)  atol=atol
    end
end

