



using JuLIP:               decode_dict
using PolyPairPots.JacobiPolys:   Jacobi

export PolyTransform, PolyCutoff1s, PolyCutoff2s, PolyTransformCut


abstract type DistanceTransform end

abstract type DistanceTransformCut <: DistanceTransform end


poly_trans(p, r0, r) = @fastmath(((1+r0)/(1+r))^p)

poly_trans_d(p, r0, r) = @fastmath((-p/(1+r0)) * ((1+r0)/(1+r))^(p+1))


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
   Dict("__id__" => "PolyPairPots_PolyTransform", "p" => T.p, "r0" => T.r0)

PolyTransform(D::Dict) = PolyTransform(D["p"], D["r0"])

convert(::Val{:PolyPairPots_PolyTransform}, D::Dict) = PolyTransform(D)

transform(t::PolyTransform, r::Number) = poly_trans(t.p, t.r0, r)

transform_d(t::PolyTransform, r::Number) = poly_trans_d(t.p, t.r0, r)



# """
# Implements the distance transform
# ```
# r -> ( (1+r0)/(1+r))^p - c0 - c1 (r - rcut)
# ```
#
# Constructor:
# ```
# PolyTransformCut(p, r0)
# ```
# """
# struct PolyTransformCut{TP, T} <: DistanceTransformCut
#    p::TP
#    r0::T
#    c0::T
#    c1::T
#    rcut::T
# end
#
# Dict(T::PolyTransformCut) =
#    Dict("__id__" => "PolyPairPots_PolyTransform",
#         "p" => T.p, "r0" => T.r0, "rcut" => rcut)
#
# PolyTransformCut(D::Dict) = PolyTransformCut(D["p"], D["r0"], D["rcut"])
#
# convert(::Val{:PolyPairPots_PolyTransformCut}, D::Dict) = PolyTransformCut(D)
#
#
# transform(t::PolyTransformCut, r::Number) =
#        (poly_trans(t.p, t.r0, r) + t.c0 + t.c1 * (r - t.rcut)) * (r < t.rcut)
#
# transform_d(t::PolyTransformCut, r::Number) =
#        (poly_trans_d(t.p, t.r0, r) + t.c1) * (r < t.rcut)
#
# function PolyTransformCut(p, r0, rcut)
#    c0 = - poly_trans(p, r0, rcut)
#    c1 = - poly_trans_d(p, r0, rcut)
#    return PolyTransformCut(p, r0, c0, c1, rcut)
# end
#
# cutoff(trans::PolyTransformCut) = trans.rcut



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
   Dict("__id__" => "PolyPairPots_PolyCutoff1s",
        "P" => P, "rl" => C.rl, "ru" => C.ru)
PolyCutoff1s(D::Dict) = PolyCutoff1s(D["P"], D["rl"],  D["ru"])
convert(::Val{:PolyPairPots_PolyCutoff1s}, D::Dict) = PolyCutoff1s(D)

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
   Dict("__id__" => "PolyPairPots_PolyCutoff2s", "P" => P,
        "rl" => C.rl, "ru" => C.ru)
PolyCutoff2s(D::Dict) = PolyCutoff2s(D["P"], D["rl"], D["ru"])
convert(::Val{:PolyPairPots_PolyCutoff2s}, D::Dict) = PolyCutoff2s(D)

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
      "__id__" => "PolyPairPots_TransformedJacobi",
      "a" => J.J.α,
      "b" => J.J.β,
      "deg" => length(J.J) - 1,
      "rl" => J.rl,
      "ru" => J.ru,
      "trans" => Dict(J.trans),
      "cutoff" => Dict(J.mult),
      "skip0" => J.J.skip0
   )

@noinline TransformedJacobi(D::Dict) =
   TransformedJacobi(
      Jacobi(D["a"], D["b"], D["deg"],
             skip0 = haskey(D, "skip0")  ? D["skip0"]  : false),
      decode_dict(D["trans"]),
      decode_dict(D["cutoff"]),
      D["rl"],
      D["ru"]
   )


Base.length(J::TransformedJacobi) = length(J.J)

cutoff(J::TransformedJacobi) = J.ru
transform(J::TransformedJacobi, r) = transform(J.trans, r)
transform_d(J::TransformedJacobi, r) = transform_d(J.trans, r)
fcut(J::TransformedJacobi, r, x) = fcut(J.mult, r, x)
fcut_d(J::TransformedJacobi, r, x) = fcut_d(J.mult, r, x)

alloc_B( J::TransformedJacobi{T}, args...) where {T} =
      Vector{T}(undef, length(J))

alloc_dB(J::TransformedJacobi{T}, args...) where {T} =
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
