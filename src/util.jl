
if VERSION < v"1.7"
    macro something(args...)
        expr = :(nothing)
        for arg in reverse(args)
            expr = :(val = $(esc(arg)); val !== nothing ? val : ($expr))
        end
        something = GlobalRef(MuseInference, :something)
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
    ẏs = pmap(tuple.(eachindex(x),step); batch_size) do (n, step)
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


# ComponentArray constructor is ridiculousy slow, this type piracy
# speeds it up for the case that comes up all the time here where the
# named tuple is not nested
function ComponentArrays.make_carray_args(nt :: NamedTuple{<:Any,<:NTuple{N,Union{Number,Vector}} where N})
    i = 1
    ax = map(nt) do v
        len = length(v)
        s = len==1 ? i : i:i+len-1
        i += len
        s
    end
    vec = reduce(vcat, values(nt))
    (vec, ComponentArrays.Axis(ax))
end

function ComponentArrays.make_carray_args(nt :: NamedTuple{<:Any,<:Tuple{Number} where N})
    ([first(nt)], ComponentArrays.Axis(map(_->1, nt)))
end

NamedTupleView(nt::NamedTuple) = nt
function NamedTupleView(cv::ComponentVector)
    tp = map(k -> getproperty(cv, k), valkeys(cv))
    unval(::Val{k}) where k = k
    NamedTuple{map(unval,valkeys(cv))}(tp)
end

LinearAlgebra.inv(A::ComponentMatrix{<:Real, <:Symmetric}) = ComponentArray(Matrix(inv(getdata(A))), getaxes(A))