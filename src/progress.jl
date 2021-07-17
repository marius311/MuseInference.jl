
# this stuff pending PR into ProgressMeter.jl, for now copied here

using Serialization
using ProgressMeter: AbstractProgress
import ProgressMeter: next!, update!, finish!, cancel

"""
    RemoteProgressWrapper(pbar::AbstractProgress) 

Wraps any `AbstractProgress` and returns another `AbstractProgress`
which is safe to ship to remote workers. Calls to `next!`, `update!`,
`cancel`, and `finish!` on the remote workers will be forwarded to the
master process and will be asynchronously applied to the original
`pbar` object.
"""
struct RemoteProgressWrapper <: AbstractProgress
    channel
    pbar
    function RemoteProgressWrapper(pbar::AbstractProgress)
        channel = RemoteChannel(()->Channel{Any}(), 1)
        @async while true
            msg = take!(channel)
            isnothing(msg) && break
            (func, args, kwargs) = msg
            func(pbar, args...; kwargs...)
        end
        new(channel, pbar)
    end
    RemoteProgressWrapper(channel::RemoteChannel) = new(channel)
end

next!(pbar::RemoteProgressWrapper, args...; kwargs...)   =  put!(pbar.channel, (next!,   args, kwargs))
update!(pbar::RemoteProgressWrapper, args...; kwargs...) =  put!(pbar.channel, (update!, args, kwargs))
finish!(pbar::RemoteProgressWrapper, args...; kwargs...) = (put!(pbar.channel, (finish!, args, kwargs)); close(pbar.channel))
cancel(pbar::RemoteProgressWrapper, args...; kwargs...)  = (put!(pbar.channel, (cancel,  args, kwargs)); close(pbar.channel))

function Serialization.serialize(s::AbstractSerializer, pbar::RemoteProgressWrapper)
    # only serializes the channel, remote workers never receive the
    # `pbar` field (which isn't needed / can't be serialized)
    Serialization.writetag(s.io, Serialization.OBJECT_TAG)
    Serialization.serialize(s, RemoteProgressWrapper)
    Serialization.serialize(s, pbar.channel)
end
function Serialization.deserialize(s::AbstractSerializer, ::Type{RemoteProgressWrapper})
    RemoteProgressWrapper(Serialization.deserialize(s))
end

RemoteProgress(args...; kwargs...) = RemoteProgressWrapper(Progress(args...; kwargs...))
RemoteProgressThresh(args...; kwargs...) = RemoteProgressWrapper(ProgressThresh(args...; kwargs...))
RemoteProgressUnknown(args...; kwargs...) = RemoteProgressWrapper(ProgressUnknown(args...; kwargs...))

