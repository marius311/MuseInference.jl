

# some convenient type-piracy for scalars
AD.gradient(ad::AD.AbstractBackend, f, x::Real) = AD.derivative(ad, f, x)
AD.hessian(ad::AD.AbstractBackend, f, x::Real) = first.(AD.hessian(ad, f∘first, [x]))

function optim_only_fg!(func, autodiff)
    Optim.only_fg!() do F, G, z
        if G != nothing
            f, g = AD.value_and_gradient(autodiff, func, z)
            G .= first(g)
            return f
        end
        if F != nothing
            return func(z)
        end
    end
end