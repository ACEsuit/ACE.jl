
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------



export PolyPairBasis

# TODO: allow PairBasis with different Pr for each z, z' combination

struct PolyPairBasis{TJ, NZ} <: IPBasis
   J::TJ
   zlist::SZList{NZ}
   bidx0::SMatrix{NZ,NZ,Int}
end

fltype(pB::PolyPairBasis) = fltype(pB.J)

Base.length(pB::PolyPairBasis) = length(pB.J) * (numz(pB) * (numz(pB) + 1)) ÷ 2
Base.length(pB::PolyPairBasis, z0::AtomicNumber) = length(pB, z2i(pB, z0))
Base.length(pB::PolyPairBasis, iz0::Integer) = length(pB.J)

zlist(pB::PolyPairBasis) = pB.zlist

function scaling(pB::PolyPairBasis, p)
   ww = zeros(Float64, length(pB))
   for iz0 = 1:numz(pB), iz = 1:numz(pB)
      idx0 = _Bidx0(pB, iz0, iz)
      for n = 1:length(pB.J)
         # TODO: very crude, can we do better?
         #       -> need a proper H2-orthogonbality?
         ww[idx0+n] = n^p
      end
   end
   return ww
end


PolyPairBasis(J::ScalarBasis, species) =
   PolyPairBasis( J, ZList(species; static=true) )

PolyPairBasis(J::ScalarBasis, zlist::SZList) =
   PolyPairBasis(J, zlist, get_bidx0(J, zlist))

function get_bidx0(J, zlist::SZList{NZ}) where {NZ}
   NJ = length(J)
   bidx0 = fill(zero(Int), (NZ, NZ))
   i0 = 0
   for i = 1:NZ, j = i:NZ
      bidx0[i,j] = i0
      bidx0[j,i] = i0
      i0 += NJ
   end
   return SMatrix{NZ, NZ, Int}(bidx0...)
end


==(B1::PolyPairBasis, B2::PolyPairBasis) =
      (B1.J == B2.J) && (B1.zlist == B2.zlist)

JuLIP.cutoff(pB::PolyPairBasis) = cutoff(pB.J)

write_dict(pB::PolyPairBasis) = Dict(
      "__id__" => "ACE_PolyPairBasis",
          "Pr" => write_dict(pB.J),
       "zlist" => write_dict(pB.zlist) )

read_dict(::Val{:SHIPs_PolyPairBasis}, D::Dict) =
   read_dict(Val{:ACE_PolyPairBasis}(), D)

read_dict(::Val{:ACE_PolyPairBasis}, D::Dict) =
      PolyPairBasis( read_dict(D["Pr"]), read_dict(D["zlist"]) )

alloc_temp(pB::PolyPairBasis, args...) = (
              J = alloc_B(pB.J),
          tmp_J = alloc_temp(pB.J)  )

alloc_temp_d(pB::PolyPairBasis, args...) =  (
             J = alloc_B( pB.J),
         tmp_J = alloc_temp(pB.J),
            dJ = alloc_dB(pB.J),
        tmpd_J = alloc_temp_d(pB.J)  )

"""
compute the zeroth index of the basis corresponding to the potential between
two species zi, zj; as precomputed in `PolyPairBasis.bidx0`
"""
_Bidx0(pB, zi, zj) = pB.bidx0[ z2i(pB, zi), z2i(pB, zj) ]
_Bidx0(pB, i::Integer, j::Integer) = pB.bidx0[ i, j ]

function energy(pB::PolyPairBasis, at::Atoms{T}) where {T}
   E = zeros(T, length(pB))
   tmp = alloc_temp(pB)
   for (i, j, R) in pairs(at, cutoff(pB))
      r = norm(R)
      evaluate!(tmp.J, tmp.tmp_J, pB.J, r)
      idx0 = _Bidx0(pB, at.Z[i], at.Z[j])
      for n = 1:length(pB.J)
         E[idx0 + n] += 0.5 * tmp.J[n]
      end
   end
   return E
end

function forces(pB::PolyPairBasis, at::Atoms{T}) where {T}
   F = zeros(JVec{T}, length(at), length(pB))
   tmp = alloc_temp_d(pB)
   for (i, j, R) in pairs(at, cutoff(pB))
      r = norm(R)
      evaluate_d!(tmp.J, tmp.dJ, tmp.tmpd_J, pB.J, r)
      idx0 = _Bidx0(pB, at.Z[i], at.Z[j])
      for n = 1:length(pB.J)
         F[i, idx0 + n] += 0.5 * tmp.dJ[n] * (R/r)
         F[j, idx0 + n] -= 0.5 * tmp.dJ[n] * (R/r)
      end
   end
   return [ F[:, iB] for iB = 1:length(pB) ]
end

function virial(pB::PolyPairBasis, at::Atoms{T}) where {T}
   V = zeros(JMat{T}, length(pB))
   tmp = alloc_temp_d(pB)
   for (i, j, R) in pairs(at, cutoff(pB))
      r = norm(R)
      evaluate_d!(tmp.J, tmp.dJ, tmp.tmpd_J, pB.J, r)
      idx0 = _Bidx0(pB, at.Z[i], at.Z[j])
      for n = 1:length(pB.J)
         V[idx0 + n] -= 0.5 * (tmp.dJ[n]/r) * R * R'
      end
   end
   return V
end
