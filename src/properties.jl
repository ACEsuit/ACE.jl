
abstract type AbstractProperty end

"""
`struct Invariant{D}` : specifies that the output of an ACE is
an invariant vector of length `D`.
"""
struct Invariant{D, T} <: AbstractProperty
   _valD::Val{D}
   _T::Type{T}
end

Base.size(::Invariant{D}) where D = (D,)
Base.zero(::Type{Invariant{D, T}}) where {D,T} = zero(SVector{D, T})

@doc raw"""
`struct EuclideanVector{D}` : specifies that the output $\varphi$ of an ACE is
`D` equivariant $\mathbb{R}^3$-vectors, i.e., $\varphi \in \mathbb{R}^{3 \times D}$
which transform under $O(3)$ as
```math
      g_Q \cdot \varphi = Q \varphi
```
"""
struct EuclideanVector{D, T} <: AbstractProperty
   _valD::Val{D}
   _T::Type{T}
end

Base.size(::EuclideanVector{D}) where D = (3, D)
Base.zero(::Type{EuclideanVector{D, T}}) where {D,T} = zero(SMatrix{3, D, T})


struct SphericalVector{L, D, LEN, T} <: AbstractProperty
   _valL::Val{L}
   _valD::Val{D}
   _vallen::Val{LEN}
   _T::Type{T}
end

Base.size(::SphericalVector{L, D, LEN}) where {L, D, LEN} = (LEN, D)
Base.zero(::Type{SphericalVector{L, D, LEN, T}}) where {L, D, LEN, T} =
      zero(SMatrix{LEN, D, T})


# TODO: Implement general case
#       learn from StaticArrays.jl how to specify dimensionality!
