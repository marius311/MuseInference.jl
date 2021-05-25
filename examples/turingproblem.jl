
using Turing
using MPMEstimate
using DynamicHMC
using FiniteDifferences
using PyPlot
using CMBLensing
using Random

##

@model function noisy_funnel(x, ::Type{T}=Float64; θ=missing) where {T}
    θ ~ Uniform(0, 10)
    z = similar(x,T)
    for i in eachindex(x)
        z[i] ~ Normal(0, exp(θ/2))
        x[i] ~ Normal(z[i], 5)
    end
    x
end

N = 5
θ = 3
Random.seed!(311)
x = identity.(noisy_funnel(fill(missing,N); θ)())

model = noisy_funnel(x)
##

chain = @time sample(model, DynamicNUTS(), 500000)

prob = TuringMPMProblem(model)
θ_MPM, σθ, history = @time mpm(prob, x, 3, nsims=500, nsteps=5)

##

close(:all)
CMBLensing.plot_kde(collect(chain["θ"][:]), boundary=(0,10))
θs = range(0,10,length=1000)
plot(θs, pdf.(Normal(θ_MPM, σθ), θs))
legend(["HMC (5min)", "MPM (5sec)"])
xlabel(L"\theta")
ylabel(L"\mathcal{P}(\theta\,|\,x)")
gcf()
##
