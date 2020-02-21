
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------



using JuLIP:               decode_dict
using SHIPs.JacobiPolys:   Jacobi

import Base:   Dict, convert, ==
import JuLIP:  cutoff

export PolyTransform, PolyCutoff1s, PolyCutoff2s, IdTransform



abstract type DistanceTransform end

abstract type DistanceTransformCut <: DistanceTransform end


poly_trans(p, r0, r) = @fastmath(((1+r0)/(1+r))^p)

poly_trans_d(p, r0, r) = @fastmath((-p/(1+r0)) * ((1+r0)/(1+r))^(p+1))

poly_trans_inv(p, r0, x) = ( (1+r0)/(x^(1/p)) - 1 )


"""
Implements the distance transform
```
r -> ( (1+r0)/(1+r))^p
```

Constructor:
```
PolyTransform(p, r0)
```
"""
struct PolyTransform{TP, T} <: DistanceTransform
   p::TP
   r0::T
end

Dict(T::PolyTransform) =
   Dict("__id__" => "SHIPs_PolyTransform", "p" => T.p, "r0" => T.r0)

PolyTransform(D::Dict) = PolyTransform(D["p"], D["r0"])

convert(::Val{:SHIPs_PolyTransform}, D::Dict) = PolyTransform(D)

transform(t::PolyTransform, r::Number) = poly_trans(t.p, t.r0, r)

transform_d(t::PolyTransform, r::Number) = poly_trans_d(t.p, t.r0, r)

inv_transform(t::PolyTransform, x::Number) = poly_trans_inv(t.p, t.r0, x)


"""
`IdTransform`: Implements the distance transform `z -> z`;
Primarily used for the z-coordinate for the EnvPairPots

Constructor: `IdTransform()`
"""
struct IdTransform <: DistanceTransform
end

Dict(T::IdTransform) =  Dict("__id__" => "SHIPs_IdTransform")
IdTransform(D::Dict) = IdTransform()
convert(::Val{:SHIPs_IdTransform}, D::Dict) = IdTransform(D)
transform(t::IdTransform, z::Number) = z
transform_d(t::IdTransform, r::Number) = one(r)
inv_transform(t::IdTransform, x::Number) = x




abstract type PolyCutoff end

"""
Implements the one-sided cutoff
```
r -> (x - xu)^p
```
Constructor:
```
PolyCutoff1s(p, rcut)
```
"""
struct PolyCutoff1s{P} <: PolyCutoff
   valP::Val{P}
   rl::Float64
   ru::Float64
   PolyCutoff1s(valP::Val{P}, rl::Real, ru::Real) where {P} = (
         ((P isa Integer) && (P > 0))  ? new{P}(valP, Float64(rl), Float64(ru))
                                       : error("P must be a positive integer") )
end

PolyCutoff1s(p::Integer, ru) = PolyCutoff1s(Val(Int(p)), 0.0, ru)
PolyCutoff1s(p::Integer, rl, ru) = PolyCutoff1s(Val(Int(p)), rl, ru)

Dict(C::PolyCutoff1s{P}) where {P} =
   Dict("__id__" => "SHIPs_PolyCutoff1s",
        "P" => P, "rl" => C.rl, "ru" => C.ru)
PolyCutoff1s(D::Dict) = PolyCutoff1s(D["P"], D["rl"],  D["ru"])
convert(::Val{:SHIPs_PolyCutoff1s}, D::Dict) = PolyCutoff1s(D)

# what happened to @pure ??? => not exported anymore
fcut(C::PolyCutoff1s{P}, r::T, x::T) where {P, T} =
      r < C.ru ? @fastmath( (1 - x)^P ) : zero(T)
fcut_d(C::PolyCutoff1s{P}, r::T, x::T) where {P, T} =
      r < C.ru ? @fastmath( - P * (1 - x)^(P-1) ) : zero(T)

"""
Implements the two-sided cutoff
```
r -> (x - xu)^p (x-xl)^p
```
Constructor:
```
PolyCutoff2s(p, rl, ru)
```
where `rl` is the inner cutoff and `ru` the outer cutoff.
"""
struct PolyCutoff2s{P} <: PolyCutoff
   valP::Val{P}
   rl::Float64
   ru::Float64
   PolyCutoff2s(valP::Val{P}, rl::Real, ru::Real) where {P} = (
         ((P isa Integer) && (P > 0))  ? new{P}(valP, Float64(rl), Float64(ru))
                                       : error("P must be a positive integer") )
end

PolyCutoff2s(p::Integer, rl, ru) = PolyCutoff2s(Val(Int(p)), rl, ru)

Dict(C::PolyCutoff2s{P}) where {P} =
   Dict("__id__" => "SHIPs_PolyCutoff2s", "P" => P,
        "rl" => C.rl, "ru" => C.ru)
PolyCutoff2s(D::Dict) = PolyCutoff2s(D["P"], D["rl"], D["ru"])
convert(::Val{:SHIPs_PolyCutoff2s}, D::Dict) = PolyCutoff2s(D)

fcut(C::PolyCutoff2s{P}, r::T, x) where {P, T} =
      C.rl < r < C.ru ? @fastmath( (1 - x^2)^P ) : zero(T)
fcut_d(C::PolyCutoff2s{P}, r::T, x) where {P, T} =
      C.rl < r < C.ru ? @fastmath( -2*P * x * (1 - x^2)^(P-1) ) : zero(T)



struct OneCutoff
   rcut::Float64
end
fcut(C::OneCutoff, r, x) = r < rcut ? one(r) : zero(r)
fcut_d(C::OneCutoff, r, x) = zero(r)


# Transformed Jacobi Polynomials
# ------------------------------
# these define the radial components of the polynomials

struct TransformedJacobi{T, TT, TM} <: SHIPs.IPBasis
   J::Jacobi{T}
   trans::TT      # coordinate transform
   mult::TM       # a multiplier function (cutoff)
   rl::T          # lower bound r
   ru::T          # upper bound r
   tl::T          #  bound t(ru)
   tu::T          #  bound t(rl)
end

==(J1::TransformedJacobi, J2::TransformedJacobi) = (
   (J1.J == J2.J) &&
   (J1.trans == J2.trans) &&
   (J1.mult == J2.mult) &&
   (J1.rl == J2.rl) &&
   (J1.ru == J2.ru) )

TransformedJacobi(J, trans, mult, rl, ru) =
   TransformedJacobi(J, trans, mult, rl, ru,
                     transform(trans, rl), transform(trans, ru) )

Dict(J::TransformedJacobi) = Dict(
      "__id__" => "SHIPs_TransformedJacobi",
      "a" => J.J.α,
      "b" => J.J.β,
      "deg" => length(J.J) - 1,
      "rl" => J.rl,
      "ru" => J.ru,
      "trans" => Dict(J.trans),
      "cutoff" => Dict(J.mult),
      "skip0" => J.J.skip0
   )

TransformedJacobi(D::Dict) =
   TransformedJacobi(
      Jacobi(D["a"], D["b"], D["deg"],
             skip0 = haskey(D, "skip0")  ? D["skip0"]  : false),
      decode_dict(D["trans"]),
      decode_dict(D["cutoff"]),
      D["rl"],
      D["ru"]
   )

convert(::Val{:SHIPs_TransformedJacobi}, D::Dict) = TransformedJacobi(D)


Base.length(J::TransformedJacobi) = length(J.J)

cutoff(J::TransformedJacobi) = J.ru
transform(J::TransformedJacobi, r) = transform(J.trans, r)
transform_d(J::TransformedJacobi, r) = transform_d(J.trans, r)
corr_transform(J::TransformedJacobi, r) =
   -1 + 2 * (transform(J.trans, r) - J.tl) / (J.tu - J.tl)

fcut(J::TransformedJacobi, r, x) = fcut(J.mult, r, x)
fcut_d(J::TransformedJacobi, r, x) = fcut_d(J.mult, r, x)

SHIPs.alloc_B( J::TransformedJacobi{T}, args...) where {T} =
      Vector{T}(undef, length(J))

SHIPs.alloc_dB(J::TransformedJacobi{T}, args...) where {T} =
      Vector{T}(undef, length(J))

function evaluate!(P, tmp, J::TransformedJacobi, r)
   N = length(J)-1
   @assert length(P) >= N+1
   # transform coordinates
   t = transform(J.trans, r)
   x = -1 + 2 * (t - J.tl) / (J.tu-J.tl)
   # evaluate the cutoff multiplier
   # the (J.tu-J.tl) / 2 factor makes the basis orthonormal
   # (just for the kick of it...)
   fc = fcut(J, r, x) * sqrt(abs(2 / (J.tu-J.tl)))
   eval_basis!(P, tmp, J, r, x, fc)
end

function eval_basis!(P, tmp, J::TransformedJacobi, r, x, fc)
   N = length(J)-1
   @assert length(P) >= N+1
   if fc == 0
      fill!(P, 0.0)
   else
      # evaluate the actual Jacobi polynomials
      evaluate!(P, nothing, J.J, x)
      for n = 1:N+1
         @inbounds P[n] *= fc
      end
   end
   return P
end

function evaluate_d!(P, dP, tmp, J::TransformedJacobi, r)
   N = length(J)-1
   @assert length(P) >= N+1
   # transform coordinates
   t = transform(J.trans, r)
   x = -1 + 2 * (t - J.tl) / (J.tu-J.tl)
   dx = (2/(J.tu-J.tl)) * transform_d(J.trans, r)
   # evaluate the cutoff multiplier and chain rule
   fc = fcut(J, r, x) * sqrt(abs(2 / (J.tu-J.tl)))
   if fc ==  0
      fill!(P, 0.0)
      fill!(dP, 0.0)
   else
      fc_d = fcut_d(J, r, x) * sqrt(abs(2 / (J.tu-J.tl)))
      # evaluate the actual Jacobi polynomials + derivatives w.r.t. x
      evaluate_d!(P, dP, nothing, J.J, x)
      for n = 1:N+1
         @inbounds p = P[n]
         @inbounds dp = dP[n]
         @inbounds P[n] = p * fc
         @inbounds dP[n] = (dp * fc + p * fc_d) * dx
      end
   end
   return dP
end



TransformedJacobi(maxdeg::Integer,
                  trans::DistanceTransform,
                  cut::PolyCutoff1s{P}) where {P} =
      TransformedJacobi( Jacobi(2*P,   0, maxdeg), trans, cut, cut.rl, cut.ru)

TransformedJacobi(maxdeg::Integer,
                  trans::DistanceTransform,
                  cut::PolyCutoff2s{P}) where {P} =
      TransformedJacobi( Jacobi(2*P, 2*P, maxdeg), trans, cut, cut.rl, cut.ru)


TransformedJacobi(maxdeg::Integer,
                  trans::DistanceTransformCut,
                  rl = 0.0) =
      TransformedJacobi( Jacobi(0, 0, maxdeg, skip0=true), trans,
                         OneCutoff(cutoff(trans)),
                         rl, cutoff(trans))




struct TransformedPolys{T, TT, TJ} <: IPBasis
   J::TJ          # the actual basis
   trans::TT      # coordinate transform
   rl::T          # lower bound r
   ru::T          # upper bound r = rcut
end

==(J1::TransformedPolys, J2::TransformedPolys) = (
   (J1.J == J2.J) &&
   (J1.trans == J2.trans) &&
   (J1.rl == J2.rl) &&
   (J1.ru == J2.ru) )

TransformedPolys(J, trans, rl, ru) =
   TransformedPolys(J, trans, rl, ru)

Dict(J::TransformedPolys) = Dict(
      "__id__" => "SHIPs_TransformedPolys",
      "J" => Dict(J.J),
      "rl" => J.rl,
      "ru" => J.ru,
      "trans" => Dict(J.trans)
   )

TransformedPolys(D::Dict) =
   TransformedPolys(
      decode_dict(D["J"]),
      decode_dict(D["trans"]),
      D["rl"],
      D["ru"]
   )

convert(::Val{:SHIPs_TransformedPolys}, D::Dict) = TransformedPolys(D)


Base.length(J::TransformedPolys) = length(J.J)

cutoff(J::TransformedPolys) = J.ru

alloc_B( J::TransformedPolys, args...) = alloc_B( J.J, args...)
alloc_dB(J::TransformedPolys, args...) = alloc_dB(J.J, args...)

function evaluate!(P, tmp, J::TransformedPolys, r)
   # transform coordinates
   t = transform(J.trans, r)
   # evaluate the actual Jacobi polynomials
   evaluate!(P, nothing, J.J, t)
   return P
end

function evaluate_d!(P, dP, tmp, J::TransformedPolys, r)
   # transform coordinates
   t = transform(J.trans, r)
   dt = transform_d(J.trans, r)
   # evaluate the actual Jacobi polynomials + derivatives w.r.t. x
   evaluate_d!(P, dP, nothing, J.J, t)
   @. dP *= dt
   return dP
end
