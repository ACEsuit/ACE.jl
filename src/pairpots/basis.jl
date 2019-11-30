

import JuLIP
using JuLIP: JVec, JMat, Atoms
using JuLIP.MLIPs: IPBasis
using LinearAlgebra: norm

using JuLIP.Potentials: ZList, SZList, z2i, i2z
using StaticArrays: SMatrix

export PolyPairBasis

struct PolyPairBasis{TJ, NZ} <: IPBasis
   J::TJ
   zlist::SZList{NZ}
   bidx0::SMatrix{NZ,NZ,Int16}
end

PolyPairBasis(maxdeg::Integer, trans::DistanceTransform, fcut::PolyCutoff) =
   PolyPairBasis(:X, maxdeg, trans, fcut)

PolyPairBasis(species, maxdeg::Integer,
              trans::DistanceTransform, fcut::PolyCutoff) =
   PolyPairBasis( TransformedJacobi(maxdeg, trans, fcut),
                  ZList(species; static=true) )

PolyPairBasis(J::TransformedJacobi, zlist::SZList) =
      PolyPairBasis(J, zlist, get_bidx0(J, zlist))

function get_bidx0(J, zlist::SZList{NZ}) where {NZ}
   NJ = length(J)
   bidx0 = fill(zero(Int16), (NZ, NZ))
   i0 = 0
   for i = 1:NZ, j = i:NZ
      bidx0[i,j] = i0
      bidx0[j,i] = i0
      i0 += NJ
   end
   return SMatrix{NZ, NZ, Int16}(bidx0...)
end

==(B1::PolyPairBasis, B2::PolyPairBasis) =
      (B1.J == B2.J) && (B1.zlist == B2.zlist)

nz(B::PolyPairBasis) = length(B.zlist)

Base.length(pB::PolyPairBasis) = length(pB.J) * (nz(pB) * (nz(pB) + 1)) ÷ 2

JuLIP.cutoff(pB::PolyPairBasis) = cutoff(pB.J)

Dict(pB::PolyPairBasis) = Dict(
      "__id__" => "PolyPairPots_PolyPairBasis",
      "J" => Dict(pB.J),
      "zlist" => pB.zlist.list )

PolyPairBasis(D::Dict) = PolyPairBasis( TransformedJacobi(D["J"]),
                                        ZList(D["zlist"]; static=true) )

convert(::Val{:PolyPairPots_PolyPairBasis}, D::Dict) = PolyPairBasis(D)



alloc_temp(pB::PolyPairBasis) = (J = alloc_B(pB.J),)
alloc_temp_d(pB::PolyPairBasis, args...) = ( J = alloc_B( pB.J),
                                            dJ = alloc_dB(pB.J) )

"""
compute the zeroth index of the basis corresponding to the potential between
two species zi, zj; as precomputed in `PolyPairBasis.bidx0`
"""
_Bidx0(pB, zi, zj) = pB.bidx0[ z2i(pB, zi), z2i(pB, zj) ]

function energy(pB::PolyPairBasis, at::Atoms{T}) where {T}
   E = zeros(T, length(pB))
   stor = alloc_temp(pB)
   for (i, j, R) in pairs(at, cutoff(pB))
      r = norm(R)
      evaluate!(stor.J, nothing, pB.J, r)
      idx0 = _Bidx0(pB, at.Z[i], at.Z[j])
      for n = 1:length(pB.J)
         E[idx0 + n] += 0.5 * stor.J[n]
      end
   end
   return E
end

function forces(pB::PolyPairBasis, at::Atoms{T}) where {T}
   F = zeros(JVec{T}, length(at), length(pB))
   stor = alloc_temp_d(pB)
   for (i, j, R) in pairs(at, cutoff(pB))
      r = norm(R)
      evaluate_d!(stor.J, stor.dJ, nothing, pB.J, r)
      idx0 = _Bidx0(pB, at.Z[i], at.Z[j])
      for n = 1:length(pB.J)
         F[i, idx0 + n] += 0.5 * stor.dJ[n] * (R/r)
         F[j, idx0 + n] -= 0.5 * stor.dJ[n] * (R/r)
      end
   end
   return [ F[:, iB] for iB = 1:length(pB) ]
end

function virial(pB::PolyPairBasis, at::Atoms{T}) where {T}
   V = zeros(JMat{T}, length(pB))
   stor = alloc_temp_d(pB)
   for (i, j, R) in pairs(at, cutoff(pB))
      r = norm(R)
      evaluate_d!(stor.J, stor.dJ, nothing, pB.J, r)
      idx0 = _Bidx0(pB, at.Z[i], at.Z[j])
      for n = 1:length(pB.J)
         V[idx0 + n] -= 0.5 * (stor.dJ[n]/r) * R * R'
      end
   end
   return V
end
