
### Manual MuseProblem

struct MuseProblem{S,Gθ,GZ} <: AbstractMuseProblem
    sample_x_z :: S
    ∇θ_logLike :: Gθ
    logLike_and_∇z_logLike :: GZ
end

function MuseProblem(sample_x_z, logLike, autodiff::ADBackend=ForwardDiffAD())
    MuseProblem(
        sample_x_z,
        (x,θ,z) -> _gradient(autodiff, θ -> logLike(x,θ,z), θ),
        (x,θ,z) -> _val_and_gradient(autodiff, z -> logLike(x,θ,z), z)
    )
end

∇θ_logLike(prob::MuseProblem, x, θ, z) = prob.∇θ_logLike(x, θ, z)
logLike_and_∇z_logLike(prob::MuseProblem, x, θ, z) = prob.logLike_and_∇z_logLike(x, θ, z)
sample_x_z(prob::MuseProblem, rng::AbstractRNG, θ) = prob.sample_x_z(rng, θ)
