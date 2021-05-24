
using MPMEstimate, Distributions

N = 9

prob = MPMProblem(
    function sample_x_z(rng, θ)
        z = rand(rng, Normal(0, θ), N)
        n = rand(rng, Normal(0, 1), N)
        x = n + z
        (;x, z)
    end,
    function logP(x, θ, z)
        -(1//2) * (sum(z.^2) / θ + sum((x .- z).^2))
    end
)

x, = prob.sample_x_z(1)

mpm(prob, x, 1, nsims=1000)

