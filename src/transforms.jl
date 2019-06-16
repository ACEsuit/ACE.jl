
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------



import JuLIP: cutoff
using SHIPs.JacobiPolys: Jacobi


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

transform(t::PolyTransform, r::Number) = @fastmath(((1+t.r0)/(1+r))^t.p)
transform_d(t::PolyTransform, r::Number) =
   @fastmath((-t.p/(1+t.r0)) * ((1+t.r0)/(1+r))^(t.p+1))

# x = (r0/r)^p
# r x^{1/p} = r0
inv_transform(t::PolyTransform, x::Number) = t.r0 / x^(1.0/t.p)

Base.Dict(t::PolyTransform) =
   Dict("__id__" => "SHIPs_PolyTransform",
        "p" => t.p,
        "r0" => t.r0)
PolyTransform(D::Dict) = PolyTransform(D["p"], D["r0"])
Base.convert(::Val{:NBodyIPs_PolyTransform}, D::Dict) = PolyTransform(D)
# hash(::BASIS, t::PolyTransform) = hash(t)


abstract type PolyCutoff end

"""
Implements the one-sided cutoff
```
r -> (x - xu)^p
```
Constructor:
```
PolyCutoff1s(p)
```
"""
struct PolyCutoff1s{P} <: PolyCutoff
   valP::Val{P}
end

PolyCutoff1s(p) = PolyCutoff1s(Val(Int(p)))

# what happened to @pure ???
fcut(::PolyCutoff1s{P}, x) where {P} = @fastmath( (1 - x)^P )
fcut_d(::PolyCutoff1s{P}, x) where {P} = @fastmath( - P * (1 - x)^(P-1) )

"""
Implements the two-sided cutoff
```
r -> (x - xu)^p (x-xl)^p
```
Constructor:
```
PolyCutoff1s(p)
```
"""
struct PolyCutoff2s{P} <: PolyCutoff
   valP::Val{P}
end

PolyCutoff2s(p) = PolyCutoff2s(Val(Int(p)))

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

TransformedJacobi(J, trans, mult, rl, ru) =
   TransformedJacobi(J, trans, mult, rl, ru,
                     transform(trans, rl), transform(trans, ru) )

Base.length(J::TransformedJacobi) = length(J.J)
cutoff(J::TransformedJacobi) = J.ru
transform(J::TransformedJacobi, r) = transform(J.trans, r)
transform_d(J::TransformedJacobi, r) = transform_d(J.trans, r)
fcut(J::TransformedJacobi, r) = fcut(J.mult, r)
fcut_d(J::TransformedJacobi, r) = fcut_d(J.mult, r)

SHIPs.alloc_B( J::TransformedJacobi{T}) where {T} = Vector{T}(undef, length(J.J))
SHIPs.alloc_dB(J::TransformedJacobi{T}) where {T} = Vector{T}(undef, length(J.J))

function eval_basis!(P, J::TransformedJacobi, r, N=length(P)-1)
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
   eval_basis!(P, J.J, x, N)
   # apply the cutoff multiplier
   fc = fcut(J, x)
   for n = 1:N+1
      @inbounds P[n] *= fc
   end
   return P
end

function eval_basis_d!(P, dP, J::TransformedJacobi, r, N=length(P)-1)
   @assert length(P) >= N+1
   # apply the cutoff
   if !(J.rl < r < J.ru)
      fill!(P, 0.0)
      fill!(dP, 0.0)
      return P, dP
   end
   # transform coordinates
   t = transform(J.trans, r)
   x = -1 + 2 * (t - J.tl) / (J.tu-J.tl)
   dx = (2/(J.tu-J.tl)) * transform_d(J.trans, r)
   # evaluate the actual Jacobi polynomials + derivatives w.r.t. x
   eval_basis_d!(P, dP, J.J, x, N)
   # apply the cutoff multiplier and chain rule
   fc = fcut(J, x)
   fc_d = fcut_d(J, x)
   for n = 1:N+1
      @inbounds p = P[n]
      @inbounds dp = dP[n]
      @inbounds P[n] = p * fc
      @inbounds dP[n] = (dp * fc + p * fc_d) * dx
   end
   return P, dP
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
