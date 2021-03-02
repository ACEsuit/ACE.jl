# --------------------------------------------------------------------------
# ACE.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


import Base: -, +, *, filter
import LinearAlgebra: norm

abstract type AbstractProperty end

@inline +(φ1::T, φ2::T) where {T <: AbstractProperty} = T( φ1.val + φ2.val )
@inline -(φ1::T, φ2::T) where {T <: AbstractProperty} = T( φ1.val - φ2.val )
@inline *(φ::T, λ::Number) where {T <: AbstractProperty} = T(φ.val * λ)
@inline norm(φ::T) where {T <: AbstractProperty} = norm(φ.val)
@inline Base.length(φ::AbstractProperty) = length(φ.val)
@inline Base.size(φ::AbstractProperty) = size(φ.val)
@inline Base.zero(φ::T) where {T <: AbstractProperty} = T(zero(φ.val))
@inline Base.zero(::Type{T}) where {T <: AbstractProperty} = zero(T())

Base.isapprox(φ1::T, φ2::T) where {T <: AbstractProperty} =
      isapprox(φ1.val, φ2.val)

"""
`struct Invariant{D}` : specifies that the output of an ACE is
an invariant scalar.
"""
struct Invariant{T} <: AbstractProperty
   val::T
end

Invariant{T}() where {T <: Number} = Invariant{T}(zero(T))

Invariant(T::DataType = Float64) = Invariant{T}()

filter(φ::Invariant, b::Array) = ( length(b) <= 1 ? true :
     iseven(sum(bi.l for bi in b)) && iszero(sum(bi.m for bi in b))  )



@doc raw"""
`struct EuclideanVector{D, T}` : specifies that the output $\varphi$ of an
ACE is an equivariant $\varphi \in \mathbb{R}^{3}$, i.e., it transform under
$O(3)$ as
```math
      \varphi \circ Q = Q \cdot \varphi,
```
where $\cdot$ denotes the standard matrix-vector product.
"""
struct EuclideanVector{T} <: AbstractProperty
   val::SVector{3, T}
end

EuclideanVector(T = Float64) = EuclideanVector(zero(SVector{3, T}))

@doc raw"""
`struct EuclideanTensor{D, T}` : specifies that the output $\varphi$ of an
ACE is an equivariant $\varphi \in \mathbb{R}^{3^D}$, where $D$ denotes the
order of the tensor. It transforms under $O(3)$ as
```math
   \varphi \circ Q =
```
 (todo - how do we write this?)

For example if $D = 1$ then this is an equivariant vector which transforms as
```math
   \varphi \circ Q = Q \varphi
```
and if $D = 2$ then $\varphi \in \mathbb{R}^{3 \times 3}$ and transforms as
```math
   \varphi \circ Q = Q \varphi Q^T
```
"""
struct EuclideanTensor{D, T, SIZE, LEN} <: AbstractProperty
   val::SArray{SIZE, T, D, LEN}
end



struct SphericalVector{L, D, LEN, T} <: AbstractProperty
   val::SVector{LEN, T}
   _valL::Val{L}
end

# L = 0 -> (0,0)
# L = 1 -> (0,0), (1,-1), (1,0), (1,1)  -> 4
# L = 3 ->  ... + 5 -> 9
# 1 + 3 + 5 + ... + 2*L+1
# = L + 2 * (1 + ... + L) = L+1 + 2 * L * (L+1) / 2 = (L+1)^2
function SphericalVector(L::Integer; T = Float64, D = 1)
   LEN = (L+1)^2
   return SphericalVector( zero(SMatrix{LEN, D, T}), Val(L) )
end

Base.zero(::SphericalVector{L, D, LEN, T}) where {L, D, LEN, T} =
      SphericalVector( zero(SMatrix{LEN, D, T}), Val{L}() )
