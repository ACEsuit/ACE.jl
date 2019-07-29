
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------



using JuLIP:               decode_dict
using SHIPs.JacobiPolys:   Jacobi

import Base:   Dict, convert, ==
import JuLIP:  cutoff

export PolyTransform, rbasis, PolyCutoff1s, PolyCutoff2s



abstract type DistanceTransform end

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


transform(t::PolyTransform, r::Number) =
      @fastmath(((1+t.r0)/(1+r))^t.p)
transform_d(t::PolyTransform, r::Number) =
      @fastmath((-t.p/(1+t.r0)) * ((1+t.r0)/(1+r))^(t.p+1))

# x = (r0/r)^p
# r x^{1/p} = r0
inv_transform(t::PolyTransform, x::Number) = t.r0 / x^(1.0/t.p)


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
   ru::Float64
end

PolyCutoff1s(p, ru) = PolyCutoff1s(Val(Int(p)), ru)

Dict(C::PolyCutoff1s{P}) where {P} =
   Dict("__id__" => "SHIPs_PolyCutoff1s", "P" => P, "ru" => C.ru)
PolyCutoff1s(D::Dict) = PolyCutoff1s(D["P"], D["ru"])
convert(::Val{:SHIPs_PolyCutoff1s}, D::Dict) = PolyCutoff1s(D)

# what happened to @pure ??? => not exported anymore
fcut(::PolyCutoff1s{P}, x) where {P} = @fastmath( (1 - x)^P )
fcut_d(::PolyCutoff1s{P}, x) where {P} = @fastmath( - P * (1 - x)^(P-1) )

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
end

PolyCutoff2s(p, rl, ru) = PolyCutoff2s(Val(Int(p)), rl, ru)

Dict(C::PolyCutoff2s{P}) where {P} =
   Dict("__id__" => "SHIPs_PolyCutoff2s", "P" => P,
        "rl" => C.rl, "ru" => C.ru)
PolyCutoff2s(D::Dict) = PolyCutoff2s(D["P"], D["rl"], D["ru"])
convert(::Val{:SHIPs_PolyCutoff2s}, D::Dict) = PolyCutoff2s(D)

fcut(::PolyCutoff2s{P}, x) where {P} = @fastmath( (1 - x^2)^P )
fcut_d(::PolyCutoff2s{P}, x) where {P} = @fastmath( -2*P * x * (1 - x^2)^(P-1) )


# Transformed Jacobi Polynomials
# ------------------------------
# these define the radial components of the polynomials

struct TransformedJacobi{T, TT, TM}
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
      "cutoff" => Dict(J.mult)
   )

TransformedJacobi(D::Dict) = TransformedJacobi(
      Jacobi(D["a"], D["b"], D["deg"]),
      decode_dict(D["trans"]),
      decode_dict(D["cutoff"]),
      D["rl"],
      D["ru"]
   )


Base.length(J::TransformedJacobi) = length(J.J)
cutoff(J::TransformedJacobi) = J.ru
transform(J::TransformedJacobi, r) = transform(J.trans, r)
transform_d(J::TransformedJacobi, r) = transform_d(J.trans, r)
fcut(J::TransformedJacobi, r) = fcut(J.mult, r)
fcut_d(J::TransformedJacobi, r) = fcut_d(J.mult, r)

SHIPs.alloc_B( J::TransformedJacobi{T}, args...) where {T} =
      Vector{T}(undef, length(J.J))
SHIPs.alloc_dB(J::TransformedJacobi{T}, args...) where {T} =
      Vector{T}(undef, length(J.J))

function eval_basis!(P, tmp, J::TransformedJacobi, r)
   N = length(J)-1
   @assert length(P) >= N+1
   # apply the cutoff
   if !(J.rl < r < J.ru)
      fill!(P, 0.0)
      return P
   end
   # transform coordinates
   t = transform(J.trans, r)
   x = -1 + 2 * (t - J.tl) / (J.tu-J.tl)
   # evaluate the actual Jacobi polynomials
   eval_basis!(P, nothing, J.J, x)
   # apply the cutoff multiplier
   fc = fcut(J, x)
   for n = 1:N+1
      @inbounds P[n] *= fc
   end
   return P
end

function eval_basis_d!(P, dP, tmp, J::TransformedJacobi, r)
   N = length(J)-1
   @assert length(P) >= N+1
   # apply the cutoff
   if !(J.rl < r < J.ru)
      fill!(P, 0.0)
      fill!(dP, 0.0)
      # return P, dP
   end
   # transform coordinates
   t = transform(J.trans, r)
   x = -1 + 2 * (t - J.tl) / (J.tu-J.tl)
   dx = (2/(J.tu-J.tl)) * transform_d(J.trans, r)
   # evaluate the actual Jacobi polynomials + derivatives w.r.t. x
   eval_basis_d!(P, dP, nothing, J.J, x)
   # apply the cutoff multiplier and chain rule
   fc = fcut(J, x)
   fc_d = fcut_d(J, x)
   for n = 1:N+1
      @inbounds p = P[n]
      @inbounds dp = dP[n]
      @inbounds P[n] = p * fc
      @inbounds dP[n] = (dp * fc + p * fc_d) * dx
   end
   return dP
end



"""
```
rbasis(maxdeg, trans, p, ru)      # with 1-sided cutoff
rbasis(maxdeg, trans, p, rl, ru)  # with 2-sided cutoff
```
"""
rbasis(maxdeg, trans, p, ru) =
   TransformedJacobi( Jacobi(p, 0, maxdeg), trans, PolyCutoff1s(p), 0.0, ru )

rbasis(maxdeg, trans, p, rl, ru) =
   TransformedJacobi( Jacobi(p, p, maxdeg), trans, PolyCutoff2s(p), rl, ru )

TransformedJacobi(maxdeg::Integer, trans::DistanceTransform,
                  cut::PolyCutoff1s{P}) where {P} =
      TransformedJacobi( Jacobi(P, 0, maxdeg), trans, cut, 0.0, cut.ru)

TransformedJacobi(maxdeg::Integer, trans::DistanceTransform,
                  cut::PolyCutoff2s{P}) where {P} =
      TransformedJacobi( Jacobi(P, P, maxdeg), trans, cut, cut.rl, cut.ru)
