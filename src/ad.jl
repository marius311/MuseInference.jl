
### Autodiff types

abstract type ADBackend end
struct ForwardDiffAD <: ADBackend end
struct ZygoteAD <: ADBackend end

_gradient(autodiff, f, x) = _val_and_gradient(autodiff, f, x)[2]
_val_and_gradient(autodiff, f, x) = error("Run `using $(string(autodiff)[1:end-4])` to enable $autodiff")

# ForwardDiff (loaded by default)
_gradient(::ForwardDiffAD, f, x::Real) = ForwardDiff.derivative(f, x)
_gradient(::ForwardDiffAD, f, x) = ForwardDiff.gradient(f, x)
_hessian(::ForwardDiffAD, f, x::Real) = first(ForwardDiff.hessian(f∘first, [x]))
_hessian(::ForwardDiffAD, f, x) = ForwardDiff.hessian(f, x)
_jacobian(::ForwardDiffAD, f, x) = ForwardDiff.jacobian(f, x)
_val_and_gradient(::ForwardDiffAD, f, x) = f(x), _gradient(ForwardDiffAD(), f, x)

# Zygote
@init @require Zygote="e88e6eb3-aa80-5325-afca-941959d7151f" begin
    function _val_and_gradient(::ZygoteAD, f, x)
        y, back = Zygote.pullback(f, x)
        y, Zygote.back(Zygote.sensitivity(y))
    end
    _jacobian(::ZygoteAD, f, x) = first(Zygote.jacobian(f, x))
end

