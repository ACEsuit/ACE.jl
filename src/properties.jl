
import Base: -, +, *, filter
import LinearAlgebra: norm

abstract type AbstractProperty end

@inline +(φ1::T, φ2::T) where {T <: AbstractProperty} = T( φ1.val + φ2.val )
@inline -(φ1::T, φ2::T) where {T <: AbstractProperty} = T( φ1.val - φ2.val )
@inline -(φ::T) where {T <: AbstractProperty} = T( -φ.val)
@inline *(φ::T, λ::Number) where {T <: AbstractProperty} = T(φ.val * λ)
@inline *(a::Union{Number, AbstractMatrix}, φ::T) where {T <: AbstractProperty} =
      T(a * φ.val)
@inline norm(φ::T) where {T <: AbstractProperty} = norm(φ.val)
@inline Base.length(φ::AbstractProperty) = length(φ.val)
@inline Base.size(φ::AbstractProperty) = size(φ.val)
@inline Base.zero(φ::T) where {T <: AbstractProperty} = T(zero(φ.val))
@inline Base.zero(::Type{T}) where {T <: AbstractProperty} = zero(T())

@inline *(A::AbstractMatrix, φ::T) where {T <: AbstractProperty} = T(A * φ.val)
# @inline *(A::StaticArrays.SArray{Tuple{3,3}, T,2,9}, φ::EuclideanVector{T}) where {T <: Number} = EuclideanVector{T}(A * φ.val)


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

rot3Dcoeffs(::Invariant, T=Float64) = Rot3DCoeffs(T)


@doc raw"""
`struct EuclideanVector{D, T}` : specifies that the output $\varphi$ of an
ACE is an equivariant $\varphi \in \mathbb{R}^{3}$, i.e., it transforms under
$O(3)$ as
```math
      \varphi \circ Q = Q \cdot \varphi,
```
where $\cdot$ denotes the standard matrix-vector product.
"""
struct EuclideanVector{T} <: AbstractProperty
   val::SVector{3, T}
end

EuclideanVector{T}() where {T <: Number} = EuclideanVector{T}(zero(SVector{3, T}))

EuclideanVector(T::DataType=Float64) = EuclideanVector{T}()

filter(φ::EuclideanVector, b::Array) = ( length(b) <= 1 ? true :
             isodd( sum(bi.l for bi in b)) &&
            (abs(sum(bi.m for bi in b)) <= 1) )

rot3Dcoeffs(::EuclideanVector,T=Float64) = Rot3DCoeffsEquiv{T,1}(Dict[], ClebschGordan(T))


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



struct SphericalVector{L, LEN, T} <: AbstractProperty
   val::SVector{LEN, T}
   _valL::Val{L} ## Why do we need Val?
end

getL(φ::SphericalVector) = typeof(φ).parameters[1];

# L = 0 -> (0,0)
# L = 1 -> (0,0), (1,-1), (1,0), (1,1)  -> 4
# L = 3 ->  ... + 5 -> 9
# 1 + 3 + 5 + ... + 2*L+1
# = L + 2 * (1 + ... + L) = L+1 + 2 * L * (L+1) / 2 = (L+1)^2
function SphericalVector(L::Integer; T = Float64)
   LEN = 2L+1   # length of SH basis up to L
   return SphericalVector( zero(SVector{LEN, T}), Val{L}() )
end

function SphericalVector{L, LEN, T}(x::AbstractArray) where {L, LEN, T}
   @assert length(x) == LEN
   SphericalVector{L, LEN, T}( SVector{LEN, T}(x...), Val(L) )
end

SphericalVector{L, LEN, T}()  where {L, LEN, T} =
      SphericalVector( zero(SVector{LEN, T}), Val{L}() )

filter(φ::SphericalVector, b::Array) = ( length(b) <= 1 ? true :
        ( ( iseven(sum(bi.l for bi in b)) == iseven(getL(φ)) ) &&
         ( abs(sum(bi.m for bi in b)) <= getL(φ) )  ) )

rot3Dcoeffs(::SphericalVector, T::DataType=Float64) = Rot3DCoeffs(T)

struct Sphericalvector{LEN, T} <: AbstractProperty
   val::SVector{LEN, T}
   #_valL::Val{L} ## Why do we need such value?
   _valL::Int64
end

getL(φ::Sphericalvector) = φ._valL

function Sphericalvector(L::Integer; T = Float64)
   LEN = 2L+1   # length of SH basis up to L
   return Sphericalvector( zero(SVector{LEN, T}), L )
end

Base.zero(::Sphericalvector{LEN, T}) where {L, LEN, T} =
      Sphericalvector( zero(SVector{LEN, T}), L )

filter(φ::Sphericalvector, b::Array) = ( length(b) <= 1 ? true :
     ( ( iseven(sum(bi.l for bi in b)) == iseven(getL(φ)) ) &&
       ( abs(sum(bi.m for bi in b)) <= getL(φ) )  ) )


# filter(φ::SphericalVector{L}, b::Array) where {L} = ( length(b) <= 1 ? true :
#              isodd( sum(bi.l for bi in b)) &&
#             (abs(sum(bi.m for bi in b)) <= 1) )
