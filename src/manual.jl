
### Manual MuseProblem

struct MuseProblem{X,S,Gθ,Pθ,GZ} <: AbstractMuseProblem
    x :: X
    sample_x_z :: S
    ∇θ_logLike :: Gθ
    logLike_and_∇z_logLike :: GZ
    logPriorθ :: Pθ
end

function MuseProblem(x, sample_x_z, logLike, logPriorθ=(θ->0), ad::AD.AbstractBackend=AD.ForwardDiffBackend())
    MuseProblem(
        x,
        sample_x_z,
        (x,θ,z) -> first(AD.gradient(ad, θ -> logLike(x,θ,z), θ)),
        (x,θ,z) -> first.(AD.value_and_gradient(ad, z -> logLike(x,θ,z), z)),
        logPriorθ
    )
end

∇θ_logLike(prob::MuseProblem, x, θ, z) = prob.∇θ_logLike(x, θ, z)
logPriorθ(prob::MuseProblem, θ) = prob.logPriorθ(θ)
logLike_and_∇z_logLike(prob::MuseProblem, x, θ, z) = prob.logLike_and_∇z_logLike(x, θ, z)
sample_x_z(prob::MuseProblem, rng::AbstractRNG, θ) = prob.sample_x_z(rng, θ)
