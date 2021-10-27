
if VERSION < v"1.7"
    macro something(args...)
        expr = :(nothing)
        for arg in reverse(args)
            expr = :(val = $(esc(arg)); val !== nothing ? val : ($expr))
        end
        something = GlobalRef(MuseEstimate, :something)
        return :($something($expr))
        end
    something() = throw(ArgumentError("No value arguments present"))
    something(x::Nothing, y...) = something(y...)
    something(x::Any, y...) = x
end


_map(args...; _...) = map(args...)

# modified version of https://github.com/JuliaDiff/FiniteDifferences.jl/blob/4d30c4389e06dd2295fd880be57bf58ca8dfc1ce/src/grad.jl#L9
# which allows 
# * specifying the step-size
# * specificying a map function (like pmap instead)
# * (parallel-friendly) progress bar
function pjacobian(f, fdm, x, step; pmap=_map, batch_size=1, pbar=nothing)
    
    x, from_vec = to_vec(x)
    ẏs = pmap(eachindex(x); batch_size) do n
        j = fdm(zero(eltype(x)), (step==nothing ? () : step)...) do ε
            xn = x[n]
            x[n] = xn + ε
            ret = copy(first(to_vec(f(from_vec(x)))))  # copy required incase `f(x)` returns something that aliases `x`
            x[n] = xn  # Can't do `x[n] -= ϵ` as floating-point math is not associative
            return ret
        end
        pbar == nothing || ProgressMeter.next!(pbar)
        return j
    end

    return (hcat(ẏs...), )

end

@init @require ComponentArrays="b0b7db55-cfe3-40fc-9ded-d10e2dbeff66" begin
    using .ComponentArrays
    function Statistics.cov(method::CovarianceEstimator, vs::AbstractVector{<:ComponentArray})
        Σ = cov(method, reduce(hcat, vs), dims=2)
        Axis = getaxes(first(vs))
        ComponentArray(Σ, (Axis..., Axis...))
    end
end