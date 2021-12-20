
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
        (x,z,θ) -> first(AD.gradient(ad, θ -> logLike(x,z,θ), θ)),
        (x,z,θ) -> first.(AD.value_and_gradient(ad, z -> logLike(x,z,θ), z)),
        logPriorθ
    )
end

∇θ_logLike(prob::MuseProblem, x, z, θ) = prob.∇θ_logLike(x, z, θ)
logPriorθ(prob::MuseProblem, θ) = prob.logPriorθ(θ)
logLike_and_∇z_logLike(prob::MuseProblem, x, z, θ) = prob.logLike_and_∇z_logLike(x, z, θ)
sample_x_z(prob::MuseProblem, rng::AbstractRNG, θ) = prob.sample_x_z(rng, θ)
