
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


import JuLIP
using JuLIP: JVec, Atoms
using JuLIP.MLIPs: IPBasis

import JuLIP: energy, forces

export PairBasis

struct PairBasis{TJ} <: IPBasis
   J::TJ
end

PairBasis(deg, trans, p, rcut) =
   PairBasis(rbasis(deg, trans, p, rcut))

bodyorder(pB::PairBasis) = 2
Base.length(pB::PairBasis) = length(pB.J)

JuLIP.cutoff(pB::PairBasis) = cutoff(pB.J)

alloc_B(pB::PairBasis) = zeros(Float64, length(pB))
alloc_dB(pB::PairBasis, N::Integer) = zeros(SVec3{Float64}, N, length_B(pB))
alloc_dB(pB::PairBasis, Rs::AbstractVector) = alloc_dB(pB, length(Rs))

alloc_temp(pB::PairBasis) = (J = alloc_B(pB.J),)
alloc_temp_d(pB::PairBasis, args...) = ( J = alloc_B( pB.J),
                                        dJ = alloc_dB(pB.J) )

function energy(at::Atoms, pB::PairBasis)
   E = alloc_B(pB)
   stor = alloc_temp(pB)
   for (i, j, r, R) in pairs(at, cutoff(pB))
      eval_basis!(stor.J, pB.J, r, nothing)
      E[:] .+= stor.J[:]
   end
   return E
end

function forces(at::Atoms, pB::PairBasis)
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
