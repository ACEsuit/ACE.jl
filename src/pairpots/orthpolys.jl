
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


module OrthPolys

using SparseArrays
import PoSH: DistanceTransform,
             transform, transform_d,
             alloc_B, alloc_dB

# struct AffineTransform{T, TT} <: DistanceTransform
#    trans::TT
#    rin::T
#    rcut::T
#    tin::T
#    tcut::T
# end
#
# AffineTransform(trans, rin, rcut) =
#       AffineTransform(trans, rin, rcut,
#                       transform(trans, rin), transform(trans, rcut))
#
# function transform(trans::AffineTransform, r)
#    t = transform(trans.trans, r)
#    return ( (t - trans.tin)  / (trans.tcut - t.tin)
#           - (t - trans.tcut) / (trans.tin - trans.tcut) )
# end
#
# function transform_d(trans::AffineTransform, r)
#    dt = transform_d(trans.trans, r)
#    return (2 / (trans.tcut - t.tin)) * dt
# end





struct OrthPolyBasis{T, TT} <: IPBasis
   # ----------------- the main polynomial parameters
   pcut::Int
   tcut::T
   A::Vector{T}
   B::Vector{T}
   C::Vector{T}
   # ----------------- used only for fitting ...
   tdf::Vector{T}
   ww::Vector{T}
end

Base.length(P::OrthPolyBasis) = length(P.A)

function OrthPolyBasis(N::Integer,
                       pcut::Integer,
                       tcut::T
                       tdf::Vector{T},
                       ww::Vector{T}) where {T <: AbstractFloat}
   @assert pcut > 0
   @assert N > 2
   A = zeros(T, N)
   B = zeros(T, N)
   C = zeros(T, N)

   # normalise the weights s.t. <1, 1> = 1
   ww = ww ./ sum(ww)
   # define inner products
   dotw = (f1, f2) -> dot(f1, ww .* f2)

   # start the iteration
   # a J1 = (t - tcut)^pcut
   _J1 = (tdf .- tcut).^pcut
   a = sqrt( dotw(_J1, _J1) )
   A[1] = 1/a
   J1 = A[1] * _J1

   # a J2 = (t - b) J1
   b = dotw(tdf .* J1, J1)
   _J2 = (tdf .- b) .* J1
   a = sqrt( dot(_J2, _J2) )
   A[2] = 1/a
   B[2] = -b / a
   J2 = (A[2] * tdf .+ B[2]) .* J1

   # keep the last two for the 3-term recursion
   Jprev = J2
   Jpprev = J1

   for n = 3:N
      # a Jn = (t - b) J_{n-1} - c J_{n-2}
      b = dotw(tdf .* Jprev, Jprev)
      c = dotw(tdf .* Jprev, Jpprev)
      _J = (tdf .- b) .* Jprev -c * Jpprev
      a = sqrt( dotw(_J, _J) )
      A[n] = 1/a
      B[n] = - b / a
      C[n] = - c / a
      Jprev, Jpprev = _J / a, Jprev
   end
end




# ------------------------------------------------------------
#

struct TransformedPolys{T, TT, TJ} <: PoSH.IPBasis
   J::TJ
   trans::TT      # coordinate transform
   rl::T          # lower bound r
   ru::T          # upper bound r
   tl::T          # bound t(ru)
   tu::T          # bound t(rl)
end

==(J1::TransformedPolys, J2::TransformedPolys) = (
   (J1.J == J2.J) &&
   (J1.trans == J2.trans) &&
   (J1.rl == J2.rl) &&
   (J1.ru == J2.ru) )

TransformedPolys(J, trans, rl, ru) =
   TransformedPolys(J, trans, rl, ru,
                    transform(trans, rl), transform(trans, ru) )

Dict(J::TransformedPolys) = Dict(
      "__id__" => "PoSH_TransformedPolys",
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

convert(::Val{:PoSH_TransformedPolys}, D::Dict) = TransformedPolys(D)


Base.length(J::TransformedPolys) = length(J.J)

cutoff(J::TransformedPolys) = J.ru
transform(J::TransformedPolys, r) = transform(J.trans, r)
transform_d(J::TransformedPolys, r) = transform_d(J.trans, r)

alloc_B( J::TransformedPolys, args...) = alloc_B( J.J, args...)
alloc_dB(J::TransformedPolys, args...) = alloc_dB(J.J, args...)

function evaluate!(P, tmp, J::TransformedPolys, r)
   if r >= J.ru
      fill!(P, 0.0)
      return P
   end
   # transform coordinates
   t = transform(J.trans, r)
   # evaluate the actual Jacobi polynomials
   evaluate!(P, nothing, J.J, t)
   return P
end

function evaluate_d!(P, dP, tmp, J::TransformedPolys, r)
   if r >= J.ru
      fill!(P, 0.0)
      fill!(dP, 0.0)
      return dP
   end
   # transform coordinates
   t = transform(J.trans, r)
   dt = transform_d(J.trans, r)
   # evaluate the actual Jacobi polynomials + derivatives w.r.t. x
   evaluate_d!(P, dP, nothing, J.J, x)
   @. dP *= dt
   return dP
end


# ----------------------------------------------------------------------
#   interface functions

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


end
