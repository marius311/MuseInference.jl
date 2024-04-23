var documenterSearchIndex = {"docs":
[{"location":"userapi/#User-API","page":"User API","title":"User API","text":"","category":"section"},{"location":"userapi/#Index","page":"User API","title":"Index","text":"","category":"section"},{"location":"userapi/","page":"User API","title":"User API","text":"Pages = [\"userapi.md\"]","category":"page"},{"location":"userapi/#Contents","page":"User API","title":"Contents","text":"","category":"section"},{"location":"userapi/","page":"User API","title":"User API","text":"muse\nMuseResult\nTuringMuseProblem","category":"page"},{"location":"userapi/#MuseInference.muse","page":"User API","title":"MuseInference.muse","text":"muse(prob::AbstractMuseProblem, θ₀; kwargs...)\nmuse!(result::MuseResult, prob::AbstractMuseProblem, [θ₀=nothing]; kwargs...)\n\nRun the MUSE estimate. The muse! form resumes an existing result. If the  muse form is used instead, θ₀ must give a starting guess for theta.\n\nSee MuseResult for description of return value. \n\nOptional keyword arguments:\n\nrng — Random number generator to use. Taken from result.rng or Random.default_rng() if not passed. \nz₀ — Starting guess for the latent space MAP.\nmaxsteps = 50 — Maximum number of iterations. \nθ_rtol = 1e-2 — Error tolerance on theta relative to its standard deviation.\n∇z_logLike_atol = 1e-2 — Absolute tolerance on the z-gradient at the MAP solution. \nnsims = 100 — Number of simulations. \nα = 0.7 — Step size for root-finder. \nprogress = false — Show progress bar.\npmap — Parallel map function. \nregularize = identity — Apply some regularization after each step. \nH⁻¹_like = nothing — Initial guess for the inverse Jacobian of s^rm MUSE(theta)\nH⁻¹_update — How to update H⁻¹_like. Should be :sims, :broyden, or :diagonal_broyden. \nbroyden_memory = Inf — How many past steps to keep for Broyden updates. \ncheckpoint_filename = nothing — Save result to a file after each iteration. \nget_covariance = false — Also call get_H and get_J to get the full covariance.\n\n\n\n","category":"function"},{"location":"userapi/#MuseInference.MuseResult","page":"User API","title":"MuseInference.MuseResult","text":"Stores the result of a MUSE run. Can be constructed by-hand as MuseResult() and passed to any of the inplace muse!, get_J!, or get_H!.\n\nFields:\n\nθ — The estimate of the theta parameters. \nΣ, Σ⁻¹ — The approximate covariance of theta and its inverse. \nH, J — The H and J matrices which form the covariance (see Millea & Seljak, 2021)\ngs — The MAP gradient sims used to compute J.\nHs — The jacobian sims used to compute H. \ndist — A Normal or MvNormal built from θ and Σ, for convenience. \nhistory — Internal diagnostic info from the run. \nrng — RNG used to generate sims for this run (so the same sims can be reused if resuming later).\ntime — Total Millisecond wall-time spent computing the result.\n\n\n\n","category":"type"},{"location":"userapi/#MuseInference.TuringMuseProblem","page":"User API","title":"MuseInference.TuringMuseProblem","text":"TuringMuseProblem(model; params = (:θ,), autodiff = Turing.ADBACKEND)\n\nWrap a Turing model to be ready to pass to muse. \n\nThe model should be conditioned on the variables which comprise the \"data\", and all other variables should be unconditioned. params should give the variable names of the parameters which MUSE will estimate. All other non-conditioned and non-params variables will be considered the latent space. E.g.,\n\n@model function demo()\n    σ ~ Normal(0, 3)\n    z ~ MvNormal(zeros(512), exp(σ/2))\n    w ~ MvNormal(z, 1)\n    x ~ MvNormal(w, 1)\n    y ~ MvNormal(w, 2)\n    (;σ,z,w,x,y)\nend\ntruth = demo()()\nmodel = demo() | (;truth.x, truth.y)\nprob = TuringMuseProblem(model, params=(:σ,))\n\nThe autodiff parameter should be either MuseInference.ForwardDiffBackend() or MuseInference.ZygoteBackend(), specifying which library to use for automatic differenation. \n\nnote: Note\nYou can also call muse, etc... directly on the model, e.g. muse(model, (σ=1,), ...), in which case the parameter names params will be read from the keys of the starting point.\n\n\n\n","category":"type"},{"location":"#MuseInference.jl","page":"MuseInference.jl","title":"MuseInference.jl","text":"","category":"section"},{"location":"","page":"MuseInference.jl","title":"MuseInference.jl","text":"(Image: ) (Image: )","category":"page"},{"location":"","page":"MuseInference.jl","title":"MuseInference.jl","text":"(Image: )","category":"page"},{"location":"","page":"MuseInference.jl","title":"MuseInference.jl","text":"The Marginal Unbiased Score Expansion (MUSE) method is a generic tool for hierarchical Bayesian inference. MUSE performs approximate marginalization over arbitrary non-Gaussian and high-dimensional latent spaces, providing Gaussianized constraints on hyper parameters of interest. It is much faster than exact methods like Hamiltonian Monte Carlo (HMC), and requires no user input like many Variational Inference (VI), and Likelihood-Free Inference (LFI) or Simulation-Based Inference (SBI) methods. It excels in high-dimensions, which challenge these other methods. It is approximate, so its results may need to be spot-checked against exact methods, but it is itself exact in asymptotic limit of a large number of data modes contributing to each hyperparameter, or in the limit of Gaussian joint likelihood regardless the number of data modes. For more details, see Millea & Seljak, 2021.","category":"page"},{"location":"","page":"MuseInference.jl","title":"MuseInference.jl","text":"MUSE works on standard hierarchical problems, where the likelihood is of the form:","category":"page"},{"location":"","page":"MuseInference.jl","title":"MuseInference.jl","text":"mathcalP(xtheta) = int rm dz  mathcalP(xztheta)","category":"page"},{"location":"","page":"MuseInference.jl","title":"MuseInference.jl","text":"In our notation, x are the observed variables (the \"data\"), z are unobserved \"latent\" variables, and theta are some \"hyperparameters\" of interest. MUSE is applicable when the goal of the analysis is to estimate the hyperparameters, theta, but otherwise, the latent variables, z, do not need to be inferred (only marginalized out via the integral above). ","category":"page"},{"location":"","page":"MuseInference.jl","title":"MuseInference.jl","text":"The only requirements to run MUSE on a particular problem are that forward simulations from mathcalP(xztheta) can be generated, and gradients of the joint likelihood, mathcalP(xztheta) with respect to z and theta can be computed. The marginal likelihood is never required, so MUSE could be considered a form of LFI/SBI. ","category":"page"},{"location":"#Install","page":"MuseInference.jl","title":"Install","text":"","category":"section"},{"location":"","page":"MuseInference.jl","title":"MuseInference.jl","text":"MuseInference.jl is a Julia package for computing the MUSE estimate. To install it, run the following from the Julia package prompt:","category":"page"},{"location":"","page":"MuseInference.jl","title":"MuseInference.jl","text":"pkg> add https://github.com/marius311/MuseInference.jl","category":"page"},{"location":"#Usage-(with-Turing.jl)","page":"MuseInference.jl","title":"Usage (with Turing.jl)","text":"","category":"section"},{"location":"","page":"MuseInference.jl","title":"MuseInference.jl","text":"The easiest way to use MuseInference is with problems defined via the Probabilistic Programming Language, Turing.jl.","category":"page"},{"location":"","page":"MuseInference.jl","title":"MuseInference.jl","text":"First, load up the relevant packages:","category":"page"},{"location":"","page":"MuseInference.jl","title":"MuseInference.jl","text":"using MuseInference, Random, Turing, Zygote, PyPlot, Printf, Dates, LinearAlgebra\nTuring.setadbackend(:zygote)\nPyPlot.ioff() # hide\nusing Logging # hide\nLogging.disable_logging(Logging.Info) # hide\nTuring.AdvancedVI.PROGRESS[] = false # hide\nTuring.PROGRESS[] = false # hide\nnothing # hide","category":"page"},{"location":"","page":"MuseInference.jl","title":"MuseInference.jl","text":"As an example, consider the following hierarchical problem, which has the classic Neal's Funnel problem embedded in it. Neal's funnel is a standard example of a non-Gaussian latent space which HMC struggles to sample efficiently without extra tricks. Specifically, we consider the model defined by:","category":"page"},{"location":"","page":"MuseInference.jl","title":"MuseInference.jl","text":"beginaligned\ntheta sim rm Normal(03)  \nz_i sim rm Normal(0exp(theta2))  \nx_i sim rm Normal(z_i 1)\nendaligned","category":"page"},{"location":"","page":"MuseInference.jl","title":"MuseInference.jl","text":"for i=1512. This problem can be described by the following Turing model:","category":"page"},{"location":"","page":"MuseInference.jl","title":"MuseInference.jl","text":"@model function funnel()\n    θ ~ Normal(0, 3)\n    z ~ MvNormal(zeros(512), exp(θ/2)*I)\n    x ~ MvNormal(z, I)\nend\nnothing # hide","category":"page"},{"location":"","page":"MuseInference.jl","title":"MuseInference.jl","text":"Next, let's choose a true value of theta=0 and generate some simulated data:","category":"page"},{"location":"","page":"MuseInference.jl","title":"MuseInference.jl","text":"Random.seed!(0)\nx = (funnel() | (θ=0,))() # draw sample of `x` to use as simulated data\nmodel = funnel() | (;x)\nnothing # hide","category":"page"},{"location":"","page":"MuseInference.jl","title":"MuseInference.jl","text":"We can run HMC on the problem to compute an \"exact\" answer to compare against:","category":"page"},{"location":"","page":"MuseInference.jl","title":"MuseInference.jl","text":"Random.seed!(0)\nsample(model, NUTS(10,0.65,init_ϵ=0.5), 10); # warmup # hide\nchain = @time sample(model, NUTS(100,0.65,init_ϵ=0.5), 500)\nnothing # hide","category":"page"},{"location":"","page":"MuseInference.jl","title":"MuseInference.jl","text":"We next compute the MUSE estimate for the same problem. To make the timing comparison fair, the number of MUSE simulations should be the same as the effective sample size of the chain we just ran. This is:","category":"page"},{"location":"","page":"MuseInference.jl","title":"MuseInference.jl","text":"nsims = round(Int, ess_rhat(chain)[:θ,:ess])","category":"page"},{"location":"","page":"MuseInference.jl","title":"MuseInference.jl","text":"Running the MUSE estimate, ","category":"page"},{"location":"","page":"MuseInference.jl","title":"MuseInference.jl","text":"muse(model, 0; nsims, get_covariance=true) # warmup # hide\nRandom.seed!(5) # hide\nmuse_result = @time muse(model, 0; nsims, get_covariance=true)\nnothing # hide","category":"page"},{"location":"","page":"MuseInference.jl","title":"MuseInference.jl","text":"Lets also try mean-field variational inference (MFVI) to compare to another approximate method.","category":"page"},{"location":"","page":"MuseInference.jl","title":"MuseInference.jl","text":"vi(model, ADVI(10, 10)) # warmup # hide\nt_vi = @time @elapsed vi_result = vi(model, ADVI(10, 1000))\nnothing # hide","category":"page"},{"location":"","page":"MuseInference.jl","title":"MuseInference.jl","text":"Now let's plot the different estimates. In this case, MUSE gives a nearly perfect answer at a fraction of the computational cost. MFVI struggles in both speed and accuracy by comparison.","category":"page"},{"location":"","page":"MuseInference.jl","title":"MuseInference.jl","text":"figure(figsize=(6,5)) # hide\naxvline(0, c=\"k\", ls=\"--\", alpha=0.5)\nhist(collect(chain[\"θ\"][:]), density=true, bins=15, label=@sprintf(\"HMC (%.1f seconds)\", chain.info.stop_time - chain.info.start_time))\nθs = range(-1,1,length=1000)\nplot(θs, pdf.(muse_result.dist, θs), label=@sprintf(\"MUSE (%.1f seconds)\", (muse_result.time / Millisecond(1000))))\nplot(θs, pdf.(Normal(vi_result.dist.m[1], vi_result.dist.σ[1]), θs), label=@sprintf(\"MFVI (%.1f seconds)\", t_vi))\nlegend()\nxlabel(L\"\\theta\")\nylabel(L\"\\mathcal{P}(\\theta\\,|\\,x)\")\ntitle(\"512-dimensional noisy funnel\")\ngcf() # hide","category":"page"},{"location":"","page":"MuseInference.jl","title":"MuseInference.jl","text":"The timing difference is indicative of the speedups over HMC that are possible. These can get even more dramatic as we increase dimensionality, which is why MUSE really excels on high-dimensional problems.","category":"page"},{"location":"#Usage-(manual)","page":"MuseInference.jl","title":"Usage (manual)","text":"","category":"section"},{"location":"","page":"MuseInference.jl","title":"MuseInference.jl","text":"It is also possible to use MuseInference without Turing. The MUSE estimate requires three things:","category":"page"},{"location":"","page":"MuseInference.jl","title":"MuseInference.jl","text":"A function which samples from the joint likelihood, mathcalP(xztheta), with signature:\nfunction sample_x_z(rng::AbstractRNG, θ)\n    # ...\n    return (;x, z)\nend\nwhere rng is an AbstractRNG object which should be used when generating random numbers, θ are the parameters, and return value should be a named tuple (;x, z). \nA function which computes the joint likelihood, mathcalP(xztheta), with signature:\nfunction logLike(x, z, θ) \n    # return log likelihood\nend\nA user-specifiable automatic differentiation library will be used to take gradients of this function. \nA function which computes the prior, mathcalP(theta), with signature:\nfunction logPrior(θ)\n    # return log prior\nend\nIf none is provided, the prior is assumed uniform. ","category":"page"},{"location":"","page":"MuseInference.jl","title":"MuseInference.jl","text":"In all cases, x, z, and θ, can be of any type which supports basic arithmetic, including scalars, Vectors, special vector types like ComponentArrays, etc...","category":"page"},{"location":"","page":"MuseInference.jl","title":"MuseInference.jl","text":"We can compute the MUSE estimate for the same funnel problem as above. To do so, first we create an MuseProblem object which specifies the three functions:","category":"page"},{"location":"","page":"MuseInference.jl","title":"MuseInference.jl","text":"prob = MuseProblem(\n    x,\n    function sample_x_z(rng, θ)\n        z = rand(rng, MvNormal(zeros(512), exp(θ/2)*I))\n        x = rand(rng, MvNormal(z, I))\n        (;x, z)\n    end,\n    function logLike(x, z, θ)\n        -(1//2) * (sum((x .- z).^2) + sum(z.^2) / exp(θ) + 512*θ)\n    end, \n    function logPrior(θ)\n        -θ^2/(2*3^2)\n    end,\n    MuseInference.ZygoteBackend()\n)\nnothing # hide","category":"page"},{"location":"","page":"MuseInference.jl","title":"MuseInference.jl","text":"And compute the estimate:","category":"page"},{"location":"","page":"MuseInference.jl","title":"MuseInference.jl","text":"Random.seed!(5) # hide\nmuse_result_manual = muse(prob, 0; nsims, get_covariance=true)\nnothing # hide","category":"page"},{"location":"","page":"MuseInference.jl","title":"MuseInference.jl","text":"This gives the same answer as before:","category":"page"},{"location":"","page":"MuseInference.jl","title":"MuseInference.jl","text":"(muse_result.θ[1], muse_result_manual.θ)","category":"page"},{"location":"devapi/#Developer-API","page":"Developer API","title":"Developer API","text":"","category":"section"},{"location":"devapi/#Index","page":"Developer API","title":"Index","text":"","category":"section"},{"location":"devapi/","page":"Developer API","title":"Developer API","text":"Pages = [\"devapi.md\"]","category":"page"},{"location":"devapi/#Contents","page":"Developer API","title":"Contents","text":"","category":"section"},{"location":"devapi/","page":"Developer API","title":"Developer API","text":"MuseInference.transform_θ\nMuseInference.inv_transform_θ\nMuseInference.sample_x_z\nMuseInference.∇θ_logLike\nMuseInference.logLike_and_∇z_logLike\nMuseInference.ẑ_at_θ\nMuseInference.standardizeθ\nMuseInference.check_self_consistency","category":"page"},{"location":"devapi/#MuseInference.transform_θ","page":"Developer API","title":"MuseInference.transform_θ","text":"transform_θ(prob::AbstractMuseProblem, θ)\n\nIf needed, custom AbstractMuseProblems should implement this to map θ to a space where its domain is (-inftyinfty).\n\n\n\n","category":"function"},{"location":"devapi/#MuseInference.inv_transform_θ","page":"Developer API","title":"MuseInference.inv_transform_θ","text":"inv_transform_θ(prob::AbstractMuseProblem, θ)\n\nIf needed, custom AbstractMuseProblems should implement this to map θ from the space where its domain is (-inftyinfty) back to the original space.\n\n\n\n","category":"function"},{"location":"devapi/#MuseInference.sample_x_z","page":"Developer API","title":"MuseInference.sample_x_z","text":"Custom AbstractMuseProblems should implement this and return a tuple (x,z) with data x and latent space z which are a sample from the joint likelihood, given θ. The signature of the function should be:\n\nsample_x_z(prob::AbstractMuseProblem, rng::AbstractRNG, θ)\n\nRandom numbers generated internally should use rng.\n\nThe θ argument to this function will always be in the un-transfored θ space.\n\n\n\n","category":"function"},{"location":"devapi/#MuseInference.∇θ_logLike","page":"Developer API","title":"MuseInference.∇θ_logLike","text":"Custom AbstractMuseProblems should implement this and return the gradient of the joint log likelihood with respect to hyper parameters θ, evaluated at data x and latent space z. The signature of the function should be:\n\n∇θ_logLike(prob::AbstractMuseProblem, x, z, θ)\n\nIf the problem needs a transformation of θ to map its domain to (-inftyinfty), then it should instead implement:\n\n∇θ_logLike(prob::AbstractMuseProblem, x, z, θ, θ_space)\n\nwhere θ_space will be either Transformedθ() or UnTransformedθ(). In this case, the θ argument will be passed in the space given by θ_space and the gradient should be w.r.t. to θ in that space.\n\nz must have domain (-inftyinfty). If a transformation is required to make this the case, that should be handled internal to this function and z will always refer to the transformed z.\n\n\n\n","category":"function"},{"location":"devapi/#MuseInference.logLike_and_∇z_logLike","page":"Developer API","title":"MuseInference.logLike_and_∇z_logLike","text":"Custom AbstractMuseProblems should implement this and return a tuple (logLike, ∇z_logLike) which give the log likelihood and its gradient with respect to the latent space z, evaluated at hyper parameters θ and data x . The signature of the function should be:\n\nlogLike_and_∇z_logLike(prob::AbstractMuseProblem, x, z, θ)\n\nz must have domain (-inftyinfty). If a transformation is required to make this the case, that should be handled internal to this function and z will always refer to the transformed z.\n\nThe θ argument to this function will always be in the un-transfored θ space.\n\nnote: Note\nAlternatively, custom problems can implement ẑ_at_θ directly and forego this method. The default ẑ_at_θ runs LBFGS with Optim.jl using logLike_and_∇z_logLike.\n\n\n\n","category":"function"},{"location":"devapi/#MuseInference.ẑ_at_θ","page":"Developer API","title":"MuseInference.ẑ_at_θ","text":"Custom AbstractMuseProblems can choose to implement this instead of logLike_and_∇z_logLike to return the best-fit latent space z given data x and parameters θ. The signature of the function should be: \n\nẑ_at_θ(prob::AbstractMuseProblem, x, z₀, θ; ∇z_logLike_atol)\n\nThe return value should be (ẑ, info) where info can be any extra diagonstic info which will be saved in the MUSE result. \n\nThe θ argument to this function will always be in the un-transfored θ space.\n\nThe z₀ should be used as a starting guess for the solution. \n\nz must have domain (-inftyinfty). If a transformation is required to make this the case, that should be handled internal to this function, and the return value should refer to the transformed z. \n\nThe default implementation of this method uses logLike_and_∇z_logLike and Optim.jl's LBFGS to iteratively maximize the log likelihood. \n\n\n\n","category":"function"},{"location":"devapi/#MuseInference.standardizeθ","page":"Developer API","title":"MuseInference.standardizeθ","text":"Custom AbstractMuseProblems can choose to implement this to turn a user-provided θ into the data-structure used internally in the computation. E.g. allow the user to pass a NamedTuple to functions like muse or get_J! while internally converting it to a ComponentVector. The signature of the function should be:\n\nstandardizeθ(prob::AbstractMuseProblem, θ)\n\n\n\n","category":"function"},{"location":"devapi/#MuseInference.check_self_consistency","page":"Developer API","title":"MuseInference.check_self_consistency","text":"check_self_consistency(\n    prob, \n    θ;\n    fdm = central_fdm(3, 1),\n    atol = 1e-3,\n    rng = Random.default_rng(),\n    has_volume_factor = true\n)\n\nChecks the self-consistency of a defined problem at a given θ, e.g. check that inv_transform_θ(prob, transform_θ(prob, θ)) ≈ θ, etc... This is mostly useful as a diagonostic when implementing a new AbstractMuseProblem. \n\nA random x and z are sampled from rng. Finite differences are computed using fdm and atol set the tolerance for ≈. has_volume_factor determines if the transformation includes the logdet jacobian in the likelihood.\n\n\n\n","category":"function"}]
}
