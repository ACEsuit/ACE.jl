using Base.Threads: threadid, nthreads
using DataStructures: Stack 

struct ArrayCache{T}
   vecs::Vector{Stack{Vector{T}}}
   mats::Vector{Stack{Matrix{T}}}
end

struct CachedArray{N, T}
   A::Array{T, N}
   pool::ArrayCache{T}
end


release!(pA::CachedArray) = release!(pA.pool, pA)

release!(A::AbstractArray) = nothing 
release!(A::Number) = nothing 

Base.getindex(pA::CachedArray, args...) = getindex(pA.A, args...)

Base.setindex!(pA::CachedArray, args...) = Base.setindex!(pA.A, args...)


function ArrayCache{T}() where {T} 
   nt = nthreads()
   vecs = [ Stack{Vector{T}}() for _=1:nt ]
   mats = [ Stack{Matrix{T}}() for _=1:nt ]
   return ArrayCache(vecs, mats)
end

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
