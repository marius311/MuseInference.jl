

# some convenient type-piracy for scalars
AD.gradient(ad::AD.AbstractBackend, f, x::Real) = AD.derivative(ad, f, x)
AD.hessian(ad::AD.AbstractBackend, f, x::Real) = first.(AD.hessian(ad, f∘first, [x]))

# ForwardDiffBackend is built-in ot AutomaticDifferentiation
@init @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" begin
    @eval const ForwardDiffBackend = AD.ForwardDiffBackend
end

# Zygote not yet in AutomaticDifferentiation, but remove this once it is
@init @require Zygote="e88e6eb3-aa80-5325-afca-941959d7151f" begin
    struct ZygoteBackend <: AD.AbstractBackend end
    AD.gradient(ad::ZygoteBackend, f, x) = Zygote.gradient(f, x)
    function AD.value_and_gradient(::ZygoteBackend, f, x)
        y, back = Zygote.pullback(f, x)
        y, back(Zygote.sensitivity(y))
    end
end
