# User API

## Summary

### Defining

Defining a problem is done in one of three ways:

* The easiest is to use the [Turing.jl](https://turing.ml) or
  [Soss.jl](https://github.com/cscherrer/Soss.jl) probabilistic
  programming languages to define a model, which is then automatically
  useable by MuseInference. See the documentation for
  [`TuringMuseProblem`](@ref) and [`SossMuseProblem`](@ref). Note that
  to use either of these, you first need run `using Turing` or `using
  Soss` in addition to `using MuseInference` in your session. 

* If the problem has $\theta$ variables whose domain is
  $(-\infty,\infty)$, and if all necessary gradients can be computed
  with automatic differentiation, you can use
  [`SimpleMuseProblem`](@ref) and specify the posterior and simulation
  generation code by-hand. 

* Otherwise, you can write a custom `AbstractMuseProblem`. See the
  [Developer API](devapi.md)


### Solving

Solving a problem is then done using:

* [`muse`](@ref) and [`muse!`](@ref) which compute the MUSE estimate
  and return a [`MuseResult`](@ref) object. The `muse!` version will
  store the result into an existing `MuseResult`, and will resume an
  existing run if the `MuseResult` already holds some work. This can
  be useful if e.g. one wishes to run the solver for more steps after
  an initial solution. Both functions accept a keyword argument
  `get_covariance` which if `true` will also compute the covariance of
  the MUSE estimate. If `false`, the covariance can be computed later
  by manually calling the following two functions:

* [`get_J!`](@ref) and [`get_H!`](@ref) which compute the $H$ and $J$
  matrices which together give the MUSE covariance, $\Sigma_{\rm
  posterior} = H^{-1}\,J\,H^{-\dagger} + \Sigma_{\rm prior}$. Calling
  these by hand rather than setting `get_covariance=true` can be
  useful as they allow more configurable options. 

### Tuning

The main tunable parameters to these functions which the user should consider are:

* `nsims = 100` — The number of simulations used. MUSE is a stochastic
  estimate. The error on the central MUSE estimate of $\theta$ scales
  as $1 / \sqrt N_{\rm sims} \cdot \sigma(\theta)$. For example, a
  MUSE solution with `nsims=100` will have roughly the same error on
  the mean as an HMC chain with an effective sample size of 100. The
  total runtime of MUSE is linear in `nsims`. A different number of
  sims can be used for `muse`, `get_J!` and `get_H!` (note that $H$ is
  generally less realization dependent and can be computed with fewer
  sims). 

* `∇z_logLike_atol = 1e-2` — The MUSE estimate involves a
  maximization of the likelihood over $z$. This controls the absolute
  solver error tolerance for $z$ for this maximization. 

* `(θ_rtol, α) = (0.1, 0.7)` — The outermost loop of the MUSE
  algorithm is solving an equation like $f(θ)=0$. The `θ_rtol`
  parameter sets the error tolerance for this solution for $\theta$,
  relative to an estimate of its uncertainty. The default `0.1` means
  the solution is good to $0.1\,\sigma_\theta$. The `α` parameter is a step-size for this solver. If a good starting guess for $\theta$ is unavailable, this may need to be reduced for the solver to converge. 

The defaults given above should give reasonable results,
but may need tweaking for individual problems. A good strategy is to
reduce `∇z_logLike_atol`, `θ_rtol`, and `α` to ensure a stable
solution, then experiment with which can be increased without screwing
up the result. The convergence properties are not largely affected by
`nsims` so this can be lowered during this experimentation to offset
the longer run-time.

### Parallelizing

MUSE is trivially parallelizable over the `nsims` simulations which
are averaged over in the algorithm. The functions `muse`, `get_J!`,
and `get_H!` can all be passed a keyword argument `pmap` which will be
used for mapping over the simulations. For example,
`pmap=Distributed.pmap` can be used to use Julia's Distributed map
across different processes. 

## Index

```@index
Pages = ["userapi.md"]
```

## Contents

```@docs
TuringMuseProblem
SossMuseProblem
SimpleMuseProblem
muse
muse!
get_J!
get_H!
MuseResult
```

