
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


module OrthPolys

using SparseArrays
using LinearAlgebra: dot

import JuLIP: evaluate!, evaluate_d!
import JuLIP.MLIPs: alloc_B, alloc_dB, IPBasis

import SHIPs
import SHIPs: DistanceTransform, transform, transform_d, TransformedPolys

import Base: ==, convert

# this is a hack to prevent a weird compiler error that I don't understand
# at all yet
___f___(D::Dict) = (@show D)

function _fcut_(pl, tl, pr, tr, t)
   if (pl > 0 && t < tl) || (pr > 0 && t > tr)
      return zero(t)
   end
   return (t - tl)^pl * (t - tr)^pr
end

function _fcut_d_(pl, tl, pr, tr, t)
   if (pl > 0 && t < tl) || (pr > 0 && t > tr)
      return zero(t)
   end
   df = 0.0
   if pl > 0; df += pl * (t - tl)^(pl-1) * (t-tr )^pr ; end
   if pr  > 0; df += pr  * (t -  tr)^(pr -1) * (t-tl)^pl; end
   return df
end

# TODO: At the moment, the cutoff is hard-coded; we could
#       weaken this again.

struct OrthPolyBasis{T} <: IPBasis
   # ----------------- the parameters for the cutoff function
   pl::Int
   tl::T
   pr::Int
   tr::T
   # ----------------- the main polynomial parameters
   A::Vector{T}
   B::Vector{T}
   C::Vector{T}
   # ----------------- used only for construction ...
   #                   but useful to have since it defines the notion or orth.
   tdf::Vector{T}
   ww::Vector{T}
end

Base.length(P::OrthPolyBasis) = length(P.A)

==(J1::OrthPolyBasis, J2::OrthPolyBasis) =
      all( getfield(J1, sym) == getfield(J2, sym)
           for sym in (:pr, :tr, :pl, :tl, :A, :B, :C) )

Dict(J::OrthPolyBasis) = Dict(
      "__id__" => "SHIPs_OrthPolyBasis",
      "pr" => J.pr,
      "tr" => J.tr,
      "pl" => J.pl,
      "tl" => J.tl,
      "A" => J.A,
      "B" => J.B,
      "C" => J.C
   )

OrthPolyBasis(D::Dict, T=Float64) =
   OrthPolyBasis(
      D["pl"], D["tl"], D["pr"], D["tr"],
      Vector{T}(D["A"]), Vector{T}(D["B"]), Vector{T}(D["C"]),
      T[], T[]
   )

convert(::Val{:SHIPs_OrthPolyBasis}, D::Dict) = OrthPolyBasis(D)

# rand applied to a J will return a random transformed distance drawn from
# the measure w.r.t. which the polynomials were constructed.
function SHIPs.rand_radial(J::OrthPolyBasis)
   @assert maximum(abs, diff(J.ww)) == 0
   return rand(J.tdf)
end

function OrthPolyBasis(N::Integer,
                       pcut::Integer,
                       tcut::T,
                       pin::Integer,
                       tin::T,
                       tdf::AbstractVector{T},
                       ww::AbstractVector{T} = ones(T, length(tdf))
                       ) where {T <: AbstractFloat}
   @assert pcut >= 0  && pin >= 0
   @assert N > 2

   if tcut < tin
      tl, tr = tcut, tin
      pl, pr = pcut, pin
   else
      tl, tr = tin, tcut
      pl, pr = pin, pcut
   end

   if minimum(tdf) < tl || maximum(tdf) > tr
      @warn("OrthoPolyBasis: t range outside [tl, tr]")
   end

   A = zeros(T, N)
   B = zeros(T, N)
   C = zeros(T, N)

   # normalise the weights s.t. <1, 1> = 1
   ww = ww ./ sum(ww)
   # define inner products
   dotw = (f1, f2) -> dot(f1, ww .* f2)

   # start the iteration
   _J1 = _fcut_.(pl, tl, pr, tr, tdf)
   a = sqrt( dotw(_J1, _J1) )
   A[1] = 1/a
   J1 = A[1] * _J1

   # a J2 = (t - b) J1
   b = dotw(tdf .* J1, J1)
   _J2 = (tdf .- b) .* J1
   a = sqrt( dotw(_J2, _J2) )
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

   return OrthPolyBasis(pl, tl, pr, tr, A, B, C, collect(tdf), collect(ww))
end

alloc_B( J::OrthPolyBasis{T}) where {T} = zeros(T, length(J))
alloc_dB(J::OrthPolyBasis{T}) where {T} = zeros(T, length(J))

alloc_B( J::OrthPolyBasis{T}, x::TX) where {T, TX} = zeros(TX, length(J))
alloc_dB(J::OrthPolyBasis{T}, x::TX) where {T, TX} = zeros(TX, length(J))

function evaluate!(P, tmp, J::OrthPolyBasis, t)
   @assert length(J) <= length(P)
   P[1] = J.A[1] * _fcut_(J.pl, J.tl, J.pr, J.tr, t)
   if length(J) == 1; return P; end
   P[2] = (J.A[2] * t + J.B[2]) * P[1]
   if length(J) == 2; return P; end
   @inbounds for n = 3:length(J)
      P[n] = (J.A[n] * t + J.B[n]) * P[n-1] + J.C[n] * P[n-2]
   end
   return P
end

function evaluate_d!(P, dP, tmp, J::OrthPolyBasis, t)
   @assert length(J) <= min(length(P), length(dP))

   P[1] = J.A[1] * _fcut_(J.pl, J.tl, J.pr, J.tr, t)
   dP[1] = J.A[1] * _fcut_d_(J.pl, J.tl, J.pr, J.tr, t)
   if length(J) == 1; return dP; end

   α = J.A[2] * t + J.B[2]
   P[2] = α * P[1]
   dP[2] = α * dP[1] + J.A[2] * P[1]
   if length(J) == 2; return dP; end

   @inbounds for n = 3:length(J)
      α = J.A[n] * t + J.B[n]
      P[n] = α * P[n-1] + J.C[n] * P[n-2]
      dP[n] = α * dP[n-1] + J.C[n] * dP[n-2] + J.A[n] * P[n-1]
   end
   return dP
end

function discrete_jacobi(N; pcut=2, tcut=1.0, pin=0, tin=-1.0, Nquad = 1000)
   tl, tr = minmax(tin, tcut)
   dt = (tr - tl) / Nquad
   tdf = range(tl + dt/2, tr - dt/2, length=Nquad)
   return OrthPolyBasis(N, pcut, tcut, pin, tin, tdf)
end

function transformed_jacobi(maxdeg::Integer,
                            trans::DistanceTransform,
                            rcut::Real, rin::Real = 0.0;
                            kwargs...)
   J =  discrete_jacobi(maxdeg; tcut = transform(trans, rcut),
                                tin = transform(trans, rin),
                                kwargs...)
   return TransformedPolys(J, trans, rin, rcut)
end



include("one_orthogonal.jl")

end
