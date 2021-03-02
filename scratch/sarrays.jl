

using StaticArrays


x = @SVector rand(3)
typeof(x.data)

struct MySVec{N, T} <: StaticVector{N, T}
   data::NTuple{N, T}
end

@inline Base.getindex(x::MySVec{N, T}, i::Int) where {N, T} = x.data[i]
StaticArrays.similar_type(x::MySVec{N, T}) where {N, T} = zero(MySVec{N, T})
StaticArrays.similar_type(VT::Type{MySVec{N, T}}) where {N, T} = zero(VT)

y = rand(MySVec{3, Float64})
z = rand(MySVec{3, Float64})


y + z

using StaticArrays

ACE.Invariant( (@SVector rand(3)) )


A2B = [ ACE.Invariant( (@SVector rand(3)) )
        for i = 1:10, j = 1:10 ]
AA = rand(10)
A2B * AA

using SparseArrays
A2Bsp = sparse(A2B)
A2Bsp * AA == A2B * AA
