
### RankReducedMPMProblem


struct RankReducedMPMProblem{M<:AbstractMPMProblem, V, A} <: AbstractMPMProblem
    prob :: M
    θ₀ :: V
    U :: A
end

∇θ_logLike(rrprob::RankReducedMPMProblem, x, θ, z) = rrprob.U' * ∇θ_logLike(rrprob.prob, x, modes_to_original_θ(rrprob, θ), z)
logLike_and_∇z_logLike(rrprob::RankReducedMPMProblem, x, θ, z) = logLike_and_∇z_logLike(rrprob.prob, x, modes_to_original_θ(rrprob, θ), z)
sample_x_z(rrprob::RankReducedMPMProblem, rng::AbstractRNG, θ) = sample_x_z(rrprob.prob, rng, modes_to_original_θ(rrprob, θ))
ẑ_at_θ(rrprob::RankReducedMPMProblem, x, θ, z₀) = ẑ_at_θ(rrprob.prob, x, modes_to_original_θ(rrprob, θ), z₀)

modes_to_original_θ(rrprob::RankReducedMPMProblem, m) = rrprob.θ₀ + rrprob.U * m