
# modified version of https://github.com/JuliaDiff/FiniteDifferences.jl/blob/4d30c4389e06dd2295fd880be57bf58ca8dfc1ce/src/grad.jl#L9
# which allows 
# * specifying the step-size
# * specificying a map function (like pmap instead)
# * (parallel-friendly) progress bar
function pjacobian(f, fdm, x, step; pmap=map, update_pbar=nothing)
    
    x, from_vec = to_vec(x)
    ẏs = pmap(eachindex(x)) do n
        j = fdm(zero(eltype(x)), (step==nothing ? () : step)...) do ε
            xn = x[n]
            x[n] = xn + ε
            ret = copy(first(to_vec(f(from_vec(x)))))  # copy required incase `f(x)` returns something that aliases `x`
            x[n] = xn  # Can't do `x[n] -= ϵ` as floating-point math is not associative
            return ret
        end
        update_pbar==nothing || update_pbar()
        return j
    end

    return (hcat(ẏs...), )

end