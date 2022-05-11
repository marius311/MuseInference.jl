# Developer API

## Summary

This page describes how to create a custom `AbstractMuseProblem` type. You might want to do this if you are creating a new interface between MuseInference and some PPL package that is not currently supported, or if you have a problem which [`SimpleMuseProblem`](@ref) cannot handle. You _do not_ need to do this if you have a model you can describe with a supported PPL package (Turing or Soss), or if [`SimpleMuseProblem`](@ref) is sufficient for you. 

As a reminder, MUSE works on joint posteriors of the form,

```math
\mathcal{P}(x,z\,|\,\theta) \mathcal{P}(\theta)
```

where $x$ represents one or more observed variables, $z$ represents one or more latent variables, and $\theta$ represents one of more hyper parameters which will be estimated by MUSE. The interface below more or less involves mapping these variables from your original problem to the form expected by MuseInference. The minimum functions you need to implement to get MUSE working are:

* [`MuseInference.sample_x_z`](@ref) to sample from $(x,z) \sim \mathcal{P}(x,z\,|\,\theta)$

* [`MuseInference.∇θ_logLike`](@ref) to compute the gradient,$\nabla_\theta \log\mathcal{P}(x,z\,|\,\theta)$.

* [`MuseInference.logLike_and_∇z_logLike`](@ref) to compute $(\log\mathcal{P}(x,z\,|\,\theta), \nabla_z \log\mathcal{P}(x,z\,|\,\theta))$.

* [`MuseInference.logPriorθ`](@ref) (_optional_) to compute the prior, $\log\mathcal{P}(\theta)$ (defaults to zero).

The $(x,z,\theta)$ can be any types which support basic arithmetic. 

Internally in the function [`MuseInference.ẑ_at_θ`](@ref), MuseInference does a maximization over $z$ using `logLike_and_∇z_logLike` and `Optim.jl`'s `LBFGS` solver. If you'd like, you can customize the entire maximization by directly implementing `ẑ_at_θ` yourself, in which case you do not need to implement `logLike_and_∇z_logLike` at all. 


MuseInference assumes $z$ and $\theta$ have support on $(-\infty,\infty)$. For some problems, this may not be the case, e.g. if you have a $\theta \sim {\rm LogNormal}$, then $\theta$ only has support on $(0,\infty)$. If this is the case for your problem, you have three options:

* If none of the internal solvers "bump up" against the edges of the support, then you don't need to do anything else.

* Outside of MuseInference, you can perform a change-of-variables for $\theta$ and/or $z$ such that the transformed variables have support on $(-\infty,\infty)$, and implement the functions above in terms of the transformed variables. In this case, MuseInference never knows (or needs to know) about the transformation, and the returned estimate of $\theta$ will be an estimate of the transformed $\theta$ (which if desired you can transform back outside of MuseInference).

* If you would like MuseInference itself to return an estimate of the _untransformed_ $\theta$, then you can implement:

    * [`MuseInference.transform_θ`](@ref)
    * [`MuseInference.inv_transform_θ`](@ref)
    * The extended forms of [`MuseInference.∇θ_logLike`](@ref) and [`MuseInference.logPriorθ`](@ref) which take a `θ_space` argument.

MuseInference doesn't provide an estimate of $z$, so if necessary, you should handle transforming it to $(-\infty,\infty)$ outside of MuseInference.

Once your define the custom `AbstractMuseProblem`, you can use [`MuseInference.check_self_consistency`](@ref) to run some self-consistency checks on it.

## Index

```@index
Pages = ["devapi.md"]
```

## Contents

```@docs
MuseInference.transform_θ
MuseInference.inv_transform_θ
MuseInference.sample_x_z
MuseInference.∇θ_logLike
MuseInference.logLike_and_∇z_logLike
MuseInference.logPriorθ
MuseInference.ẑ_at_θ
MuseInference.standardizeθ
MuseInference.check_self_consistency
```

