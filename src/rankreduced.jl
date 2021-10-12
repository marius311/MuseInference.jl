
### RankReducedMuseProblem


struct RankReducedMuseProblem{M<:AbstractMuseProblem, V, A} <: AbstractMuseProblem
    prob :: M
    θ₀ :: V
    U :: A
end

∇θ_logLike(rrprob::RankReducedMuseProblem, x, θ, z) = rrprob.U' * ∇θ_logLike(rrprob.prob, x, modes_to_original_θ(rrprob, θ), z)
logLike_and_∇z_logLike(rrprob::RankReducedMuseProblem, x, θ, z) = logLike_and_∇z_logLike(rrprob.prob, x, modes_to_original_θ(rrprob, θ), z)
sample_x_z(rrprob::RankReducedMuseProblem, rng::AbstractRNG, θ) = sample_x_z(rrprob.prob, rng, modes_to_original_θ(rrprob, θ))
ẑ_at_θ(rrprob::RankReducedMuseProblem, x, θ, z₀; kwargs...) = ẑ_at_θ(rrprob.prob, x, modes_to_original_θ(rrprob, θ), z₀; kwargs...)

modes_to_original_θ(rrprob::RankReducedMuseProblem, m) = rrprob.θ₀ + rrprob.U * m



struct ReparameterizedMuseProblem{M<:AbstractMuseProblem} <: AbstractMuseProblem
    prob :: M
    f
    f⁻¹
end

∇θ_logLike(rprob::ReparameterizedMuseProblem, x, θ, z) = 
    convert(typeof(θ), _jacobian(ForwardDiffAD(), rprob.f, θ)' * ∇θ_logLike(rprob.prob, x, rprob.f(θ), z))
logLike_and_∇z_logLike(rprob::ReparameterizedMuseProblem, x, θ, z) = logLike_and_∇z_logLike(rprob.prob, x, rprob.f(θ), z)
sample_x_z(rprob::ReparameterizedMuseProblem, rng::AbstractRNG, θ) = sample_x_z(rprob.prob, rng, rprob.f(θ))
ẑ_at_θ(rprob::ReparameterizedMuseProblem, x, θ, z₀; kwargs...) = ẑ_at_θ(rprob.prob, x, rprob.f(θ), z₀; kwargs...)
