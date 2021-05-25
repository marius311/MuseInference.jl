
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
