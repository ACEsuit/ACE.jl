
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

import PoSH: DistanceTransform, transform, transform_d

import Base: ==, convert

# this is a hack to prevent a weird compiler error that I don't understand yet
___f___(D::Dict) = (@show D)


struct OrthPolyBasis{T} <: IPBasis
   # ----------------- the main polynomial parameters
   pcut::Int
   tcut::T
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
           for sym in (:pcut, :tcut, :A, :B, :C) )

Dict(J::OrthPolyBasis) = Dict(
      "__id__" => "PoSH_OrthPolyBasis",
      "pcut" => J.pcut,
      "tcut" => J.tcut,
      "A" => J.A,
      "B" => J.B,
      "C" => J.C
   )

OrthPolyBasis(D::Dict, T=Float64) =
   OrthPolyBasis(
      D["pcut"], D["tcut"],
      Vector{T}(D["A"]), Vector{T}(D["B"]), Vector{T}(D["C"]),
      T[], T[]
   )

convert(::Val{:PoSH_OrthPolyBasis}, D::Dict) = OrthPolyBasis(D)


function OrthPolyBasis(N::Integer,
                       pcut::Integer,
                       tcut::T,
                       tdf::AbstractVector{T},
                       ww::AbstractVector{T} = ones(T, length(tdf))
                       ) where {T <: AbstractFloat}
   @assert pcut >= 0
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

   return OrthPolyBasis(pcut, tcut, A, B, C, collect(tdf), collect(ww))
end

alloc_B( J::OrthPolyBasis{T}, args...) where {T} = zeros(T, length(J))
alloc_dB(J::OrthPolyBasis{T}, args...) where {T} = zeros(T, length(J))

function evaluate!(P, tmp, J::OrthPolyBasis, t)
   P[1] = J.A[1] * (t - J.tcut)^(J.pcut)
   P[2] = (J.A[2] * t + J.B[2]) * P[1]
   for n = 3:length(J)
      P[n] = (J.A[n] * t + J.B[n]) * P[n-1] + J.C[n] * P[n-2]
   end
   return P
end

function evaluate_d!(P, dP, tmp, J::OrthPolyBasis, t)
   P[1] = J.A[1] * (t - J.tcut)^(J.pcut)
   dP[1] = ( J.pcut == 0 ? 0 :
             J.A[1] * J.pcut * (t - J.tcut)^(J.pcut - 1) )

   α = J.A[2] * t + J.B[2]
   P[2] = α * P[1]
   dP[2] = α * dP[1] + J.A[2] * P[1]

   for n = 3:length(J)
      α = J.A[n] * t + J.B[n]
      P[n] = α * P[n-1] + J.C[n] * P[n-2]
      dP[n] = α * dP[n-1] + J.C[n] * dP[n-2] + J.A[n] * P[n-1]
   end
   return dP
end


function discrete_jacobi(N; pcut=2, tcut=1.0, Nquad = 1000)
   dt = 2 / Nquad
   tdf = range(-1 + dt/2, 1 - dt/2, length=Nquad)
   return OrthPolyBasis(N, pcut, tcut, tdf)
end


end
