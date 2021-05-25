# This is copy-pasted from https://github.com/tpapp/ObjectPools.jl
# with the idea of starting to modify it and adapt it to ACE.jl use-case.
# if this doesn't get developed further, then should delete this branch to
# forcibly remove this again from the repository.
module ObjectPools

export new!, recycle!, ArrayPool

using ArgCheck: @argcheck
using Parameters: @unpack
using DocStringExtensions: SIGNATURES, TYPEDEF

####
#### general interface
####

"""
    new!(pool, specifications...)

Return a yet unused object from `pool` according to `specifications`. The interpretation of
specifications depends on the pool.
"""
function new! end

####
#### specific pools
####

"""
$(TYPEDEF)

This type is used for internal implementation only.
"""
mutable struct HomogeneousArrayPool{T,N,D}
    arrays::Vector{Array{T,N}}
    used::Int
    """
    $(SIGNATURES)

    An object pool for homogeneous arrays with element type `T` and dimensions `D`.
    """
    function HomogeneousArrayPool{T}(D::NTuple{N,Int}) where {T,N}
        @argcheck all(d -> d isa Int && d ≥ 0, D)
        new{T,N,D}(Vector{Array{T,N}}(), 0)
    end
end

"""
$(SIGNATURES)

A new array of the type and dimensions determined by the pool.
"""
function _new!(pool::HomogeneousArrayPool{T,N,D}) where {T,N,D}
    @unpack arrays = pool
    pool.used += 1
    if length(arrays) ≥ pool.used
        arrays[pool.used]
    else
        A = Array{T}(undef, D)
        push!(arrays, A)
        A
    end
end

function _recycle!(pool::HomogeneousArrayPool)
    pool.used = 0
    nothing
end

struct ArrayPool
    arrays::Dict{Any,Any}
    @doc """
    $(SIGNATURES)

    An object pool of `Array`s with heterogeneous element types and dimensions.
    """
    ArrayPool() = new(Dict{Any,Any}())
end

"""
    new!(pool, Array{T}, dims)
    new!(pool, Array{T}, dims...)

A new `Array{T}` with dimensions `dims` (can also be specified in a splat syntax).
"""
function new!(pool::ArrayPool, ::Type{<:Array{T,N}}, dims::NTuple{N,Int}) where {T,N}
    _pool = get!(() -> HomogeneousArrayPool{T}(dims), pool.arrays, (T, dims...))
    _new!(_pool)::Array{T,N}
end

new!(pool::ArrayPool, T, dims::Int...) = new!(pool, T, dims)

function new!(pool::ArrayPool, ::Type{Array{T}}, dims::NTuple{N,Int}) where {T,N}
    new!(pool, Array{T,N}, dims)
end

function recycle!(pool::ArrayPool)
    foreach(_recycle!, values(pool.arrays))
    nothing
end

end # module
