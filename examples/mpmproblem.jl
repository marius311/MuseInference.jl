
using MPMEstimate, Distributions, Random

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


x, = prob.sample_x_z(MersenneTwister(1), 1)

θ_MPM, σθ, history = mpm(prob, x, 1, nsims=1000, nsteps=10)

##
close(:all)
plot([h.θ for h in history])
gcf()
##