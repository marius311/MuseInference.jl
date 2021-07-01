
# 
function jacobian(f, fdm, x, step...)
    x, from_vec = to_vec(x)
    @everywhere cuda_gc()
    ẏs = @showprogress pmap(eachindex(x)) do n
        return fdm(zero(eltype(x)), step...) do ε
            xn = x[n]
            x[n] = xn + ε
            ret = copy(first(to_vec(f(from_vec(x)))))  # copy required incase `f(x)` returns something that aliases `x`
            x[n] = xn  # Can't do `x[n] -= ϵ` as floating-point math is not associative
            return ret
        end
    end
    return (hcat(ẏs...), )
end