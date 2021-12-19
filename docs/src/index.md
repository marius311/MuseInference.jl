# MuseEstimate.jl

[![](https://img.shields.io/badge/documentation-latest-blue.svg)](https://cosmicmar.com/MuseEstimate.jl/latest) [![](https://img.shields.io/badge/source-github-blue)](https://github.com/marius311/MuseEstimate.jl)

[![](https://github.com/marius311/MuseEstimate.jl/actions/workflows/docs.yml/badge.svg)](https://github.com/marius311/MuseEstimate.jl/actions/workflows/docs.yml)

The Marginal Unbiased Score Expansion (MUSE) estimate is a generic tool for hierarchical Bayesian inference. It provides an (often extremely good) approximation to the posterior distribution, and is faster than other methods such as Hamiltonian Monte Carlo (HMC), Variational Inference (VI), Likelihood-Free Inference (LFI). It excels on problems which are high-dimensional and mildly-to-moderately non-Gaussian. 

MUSE works on standard hierarchical problems, where the likelihood is of the form:

```math
\mathcal{P}(x\,|\,\theta) = \int {\rm d}z \, \mathcal{P}(x\,|\,z,\theta) \, \mathcal{P}(z\,|\,\theta)
```

In our notation, $x$ are the observed variables (the "data"), $z$ are unobserved "latent" variables, and $\theta$ are some "hyperparameters" of interest. MUSE is applicable when the goal of the analysis is to estimate the hyperparameters, $\theta$, but otherwise, the latent variables, $z$, do not need to be inferred (only marginalized out via the integral above).

HMC performs the above integral via Monte Carlo, VI perfoms it by parameterizing the integrand with a function that has a known solution, and LFI performs it by importance sampling. MUSE differs in that performs it with a semi-analytic approximation which implicilty accounts for *some* of the non-Gaussianity of the integrand, and which is fast to compute. The output of the MUSE procedure is an estimate of the hyperparameters, $\hat \theta$, which can be proven to be asymptotically unbiased, and its covariance, $\Sigma_\theta$. In the asymptotic limit (i.e. a large enough data vector), and assuming uniform priors on $\theta$, this estimator and its covariance will be the same as the posterior mean and the posterior covariance of $\theta$. Even for finite data, the MUSE estimate can be considered an approximation to the Bayesian posterior, and is often sufficiently close to yield an acceptable answer for a fraction of the computational cost of the exact solution. For more details see [Millea & Seljak, 2021](http://arxiv.org/inprep).

# Install

MuseEstimate.jl is a Julia package for computing the MUSE estimate. To install it, run the following from the Julia package prompt:

```
pkg> add https://github.com/marius311/MuseEstimate.jl
```

# Usage (with Turing.jl)

The easiest way to use MuseEstimate is with problems defined via the Probabilistic Programming Language, [Turing.jl](https://turing.ml/stable/).

First, load up the relevant packages:

```@example 1
using MuseEstimate, Random, Turing, PyPlot
PyPlot.ioff() # hide
nothing # hide
```

As an example, consider the following hierarchical problem, which has the classic [Neal's Funnel](https://mc-stan.org/docs/2_18/stan-users-guide/reparameterization-section.html) problem embedded in it. Neal's funnel is a standard example of a non-Gaussian latent space which HMC struggles to sample efficiently without extra tricks. Specifically, we consider the model defined by:

```math
\theta \sim {\rm Normal(0,3)} \\ 
z_i \sim {\rm Normal}(0,\exp(\theta/2)) \\ 
x_i \sim {\rm Normal}(z_i, 1)
```

where $i=1...50$. This problem can be described by the following Turing model:
```@example 1
@model function funnel()
    θ ~ Normal(0, 3)
    z ~ MvNormal(zeros(50), exp(θ/2))
    x ~ MvNormal(z, 1)
end
nothing # hide
```

Note that, because the MUSE algorithm needs to be able to sample from $\mathcal{P}(x\,|\,\theta)$, `θ` must appear as an explicit argument to the model function, and it should have the default value of `missing`, as above.

Next, lets choose a true value of $\theta=1$ and generate some simulated data:

```@example 1
Random.seed!(0)
x = funnel()() # draw sample of `x` to use as simulated data
model = funnel() | (;x)
nothing # hide
```

We can run HMC on the problem to compute an "exact" answer to compare against:

```@example 1
Turing.PROGRESS[] = false # hide
Random.seed!(1) # hide
sample(model, NUTS(), 10); # warmup # hide
Random.seed!(1) # hide
chain = @time sample(model, NUTS(), 5000)
nothing # hide
```

And we can compute the MUSE estimate for the same problem:

```@example 1
muse(model, 1) # warmup # hide
Random.seed!(5) # hide
result = @time muse(model, 1.)
get_J!(result, model)
get_H!(result, model)
nothing # hide
```

For a more careful comparison of the two approaches in terms of the number of model gradient evaluations, see [Millea & Seljak, 2021](http://arxiv.org/inprep), but the timing difference above is indicative of the type of speedups which are possible, and the relative speedup generally increases for higher-dimensional latent spaces. 


Finally, we can compare the two estimates, verifying that in this case, MUSE gives a near exact answer:

```@example 1
figure(figsize=(6,5)) # hide
axvline(1, c="k", ls="--", alpha=0.5)
hist(collect(chain["θ"][:]), density=true, bins=20, label="HMC")
θs = range(-1,3,length=1000)
plot(θs, pdf.(Normal(result.θ, result.σθ), θs), label="MUSE")
legend()
xlabel(L"\theta")
ylabel(L"\mathcal{P}(\theta\,|\,x)")
gcf() # hide
```

### Advanced

!!! note

    Currently names `x` and `θ` are hardcoded, and there can only be one of them (although they can be vectors). Need to make is to that there can be multiple latent and hyperparameters, and then describe that here. 


# Usage (manual)

It is also possible to use MuseEstimate without Turing. The MUSE estimate requires two things:

1. A function which samples from $\mathcal{P}(x,z\,|\,\theta)$, with signature:

   ```julia
   function sample_x_z(rng::AbstractRNG, θ)
       # ...
       return (;x, z)
   end
   ```

   where `rng` is an `AbstractRNG` object which should be used when generating random numbers, `θ` are the parameters, and return value should be a named tuple `(;x, z)`. 
    
2. A function which computes the likelihood, $\mathcal{P}(x\,|\,\theta,z)$, with signature:

   ```julia
   function logLike(x, θ, z) 
       # return log likelihood
   end
   ```

   A user-specifiable automatic differentiation library will be used to take gradients of this function. 

In both (1) and (2), `x`, `θ`, and `z` can be of any type which supports basic arithmetic, including scalars, `Vector`s, special vector types like `ComponentArray`s, etc...

We can compute the MUSE estimate for the same funnel problem as above. To do so, first we create an `MuseProblem` object which specifies the two functions:

```@example 1
prob = MuseProblem(
    function sample_x_z(rng, θ)
        z = rand(rng, MvNormal(zeros(50), exp(θ/2)))
        x = rand(rng, MvNormal(z, 1))
        (;x, z)
    end,
    function logLike(x, θ, z)
        -(1//2) * (sum((x .- z).^2) + sum(z.^2) / exp(θ) + 50*θ)
    end
)
nothing # hide
```

And compute the estimate:

```@example 1
muse(prob, x, 1) # warmup # hide
Random.seed!(5) # hide
result′ = @time muse(prob, x, 1)
get_J!(result′, model)
get_H!(result′, model)
nothing # hide
```

Finally, we can verify that the answer is identical to the answer computed when the problem was specified with Turing:

```@example 1
figure(figsize=(6,5)) # hide
hist(collect(chain["θ"][:]), density=true, bins=20, label="HMC")
θs = range(-1,3,length=1000)
plot(θs, pdf.(Normal(result.θ,  result.σθ),  θs), label="MUSE (Turing)")
plot(θs, pdf.(Normal(result′.θ, result′.σθ), θs), label="MUSE (Manual)", ls="--")
legend()
xlabel(L"\theta")
ylabel(L"\mathcal{P}(\theta\,|\,x)")
gcf() # hide
```