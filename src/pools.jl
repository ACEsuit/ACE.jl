using Base.Threads: threadid, nthreads
using DataStructures: Stack 

struct ArrayCache{T}
   vecs::Vector{Stack{Vector{T}}}
   mats::Vector{Stack{Matrix{T}}}
end

struct CachedArray{N, T} <: AbstractArray{T, N} 
   A::Array{T, N}
   pool::ArrayCache{T}
end


release!(A::Any) = nothing 
release!(pA::CachedArray) = release!(pA.pool, pA)

using Base: @propagate_inbounds

@propagate_inbounds function Base.getindex(pA::CachedArray, I...) 
   @boundscheck checkbounds(pA.A, I...)
   @inbounds pA.A[I...]
end

@propagate_inbounds function Base.setindex!(pA::CachedArray, val, I...)
   @boundscheck checkbounds(pA.A, I...)
   @inbounds pA.A[I...] = val
end

# Base.getindex(pA::CachedArray, args...) = getindex(pA.A, args...)

# Base.setindex!(pA::CachedArray, args...) = setindex!(pA.A, args...)

Base.length(pA::CachedArray) = length(pA.A)

Base.eltype(pA::CachedArray) = eltype(pA.A)

Base.size(pA::CachedArray, args...) = size(pA.A, args...)

Base.parent(pA::CachedArray) = pA.A


function ArrayCache{T}() where {T} 
   nt = nthreads()
   vecs = [ Stack{Vector{T}}() for _=1:nt ]
   mats = [ Stack{Matrix{T}}() for _=1:nt ]
   return ArrayCache(vecs, mats)
end

acquire!(c::ArrayCache{T}, len::Integer, ::Type{T}) where {T} = 
         acquire!(c, len)

acquire!(c::ArrayCache{T}, len::Integer, ::Type{S}) where {T, S} =
         Vector{S}(undef, len)

function acquire!(c::ArrayCache{T}, len::Integer) where {T}
   stack = c.vecs[threadid()]
   if isempty(stack)
      A = Vector{T}(undef, len)
   else 
      A = pop!(stack)
      resize!(A, len)
   end
   return CachedArray(A, c)
end

release!(c::ArrayCache, cA::CachedArray{1}) = 
      push!(c.vecs[threadid()], cA.A)
