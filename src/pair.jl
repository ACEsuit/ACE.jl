
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


import JuLIP
using JuLIP: JVec, Atoms
using JuLIP.MLIPs: IPBasis

import JuLIP:             energy, forces, virial
import JuLIP.Potentials:  evaluate, evaluate_d
import Base: Dict, convert, ==

export PairBasis

struct PairBasis{TJ} <: IPBasis
   J::TJ
end

PairBasis(deg, trans, p, rcut) =
   PairBasis(rbasis(deg, trans, p, rcut))

==(B1::PairBasis, B2::PairBasis) = (B1.J == B2.J)

bodyorder(pB::PairBasis) = 2
Base.length(pB::PairBasis) = length(pB.J)

JuLIP.cutoff(pB::PairBasis) = cutoff(pB.J)

Dict(pB::PairBasis) = Dict(
      "__id__" => "SHIPs_PairBasis",
      "J" => Dict(pB.J) )

PairBasis(D::Dict) = PairBasis(TransformedJacobi(D["J"]))

convert(::Val{:SHIPs_PairBasis}, D::Dict) = PairBasis(D)


alloc_B(pB::PairBasis) = zeros(Float64, length(pB))
alloc_dB(pB::PairBasis, N::Integer) = zeros(SVec3{Float64}, N, length_B(pB))
alloc_dB(pB::PairBasis, Rs::AbstractVector) = alloc_dB(pB, length(Rs))

alloc_temp(pB::PairBasis) = (J = alloc_B(pB.J),)
alloc_temp_d(pB::PairBasis, args...) = ( J = alloc_B( pB.J),
                                        dJ = alloc_dB(pB.J) )

function energy(pB::PairBasis, at::Atoms)
   E = alloc_B(pB)
   stor = alloc_temp(pB)
   for (i, j, r, R) in pairs(at, cutoff(pB))
      eval_basis!(stor.J, pB.J, r, nothing)
      E[:] .+= stor.J[:]
   end
   return E
end

function forces(pB::PairBasis, at::Atoms)
   F = zeros(JVecF, length(at), length(pB))
   stor = alloc_temp_d(pB)
   for (i, j, r, R) in pairs(at, cutoff(pB))
      eval_basis_d!(stor.J, stor.dJ, pB.J, r, nothing)
      for iB = 1:length(pB)
         F[i, iB] += stor.dJ[iB] * (R/r)
         F[j, iB] -= stor.dJ[iB] * (R/r)
      end
   end
   return [ F[:, iB] for iB = 1:length(pB) ]
end

function virial(pB::PairBasis, at::Atoms)
   V = zeros(JMatF, length(pB))
   stor = alloc_temp_d(pB)
   for (i, j, r, R) in pairs(at, cutoff(pB))
      eval_basis_d!(stor.J, stor.dJ, pB.J, r, nothing)
      for iB = 1:length(pB)
         V[iB] -= stor.dJ[iB]/r * R * R'
      end
   end
   return V
end


# ----------------------------------------------------

struct PolyPairPot{T,TJ} <: AbstractCalculator
   J::TJ
   coeffs::Vector{T}
end

@pot PolyPairPot

PolyPairPot(pB::PairBasis, coeffs::Vector) = PolyPairPot(pB.J, coeffs)
JuLIP.MLIPs.combine(pB::PairBasis, coeffs::AbstractVector) = PolyPairPot(pB, coeffs)

JuLIP.cutoff(V::PolyPairPot) = cutoff(V.J)
bodyorder(V::PolyPairPot) = 2

==(V1::PolyPairPot, V2::PolyPairPot) = (
      (V1.J == V2.J) && (V1.coeffs == V2.coeffs) )

Dict(V::PolyPairPot) = Dict(
      "__id__" => "SHIPs_PolyPairPot",
      "J" => Dict(V.J),
      "coeffs" => V.coeffs)

PolyPairPot(D::Dict) = PolyPairPot(
      TransformedJacobi(D["J"]),
      Vector{Float64}(D["coeffs"]))

convert(::Val{:SHIPs_PolyPairPot}, D::Dict) = PolyPairPot(D)


alloc_temp(V::PolyPairPot) = (J = alloc_B(V.J),)
alloc_temp_d(V::PolyPairPot, args...) = ( J = alloc_B( V.J),
                                           dJ = alloc_dB(V.J) )

evaluate(V::PolyPairPot, r) =
      dot(V.coeffs, eval_basis(V.J, r))
evaluate_d(V::PolyPairPot, r) =
      dot(V.coeffs, eval_basis_d(V.J, r)[2])


function energy(V::PolyPairPot{T}, at::Atoms) where {T}
   E = zero(T)
   stor = alloc_temp(V)
   for (i, j, r, R) in pairs(at, cutoff(V))
      eval_basis!(stor.J, V.J, r, nothing)
      E += dot(V.coeffs, stor.J)
   end
   return E
end


function forces(V::PolyPairPot{T}, at::Atoms) where {T}
   F = zeros(JVec{T}, length(at))
   stor = alloc_temp_d(V)
   for (i, j, r, R) in pairs(at, cutoff(V))
      eval_basis_d!(stor.J, stor.dJ, V.J, r, nothing)
      dJ = dot(V.coeffs, stor.dJ)
      F[i] += dJ * (R/r)
      F[j] -= dJ * (R/r)
   end
   return F
end


function virial(V::PolyPairPot{T}, at::Atoms) where {T}
   V = JMat{T}
   stor = alloc_temp_d(V)
   for (i, j, r, R) in pairs(at, cutoff(V))
      eval_basis_d!(stor.J, stor.dJ, V.J, r, nothing)
      dJ = dot(V.coeffs, stor.dJ)
      V -= dJ/r * R * R'
   end
   return V
end
