
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


# k : r-degree
# l : θ-degree
# m : z-degree

import PoSH: alloc_B, alloc_dB
using PoSH: AList

import JuLIP: evaluate!, evaluate_d!,
              alloc_temp, alloc_temp_d


struct EnvPairBasis{T0, TR, TZ, TT}
   P0::T0                # basis for the bond-length coordinate
   Pr::TR                # specifies the radial basis  / n
   Pθ::TT                # the angular basis           / l
   Pz::TZ                # specifies the z-basis       / m
   alist::BondAList         # datastructure to assemble A
   aalist::BondAAList       # datastructure to assemble AA
end

Base.eltype(basis::EnvPairBasis) = eltype(basis.Pr)

alloc_B(basis::EnvPairBasis) = zeros(eltype(basis.Pr), length(basis))

alloc_temp(basis::EnvPairBasis) =
   ( Pr = alloc_B(basis.Pr),
     tmp_Pr = alloc_temp(basis.Pr),
     Pθ = alloc_B(basis.Pθ),
     tmp_Pθ = alloc_temp(basis.Pθ),
     Pz = alloc_B(basis.Pz),
     tmp_Pz = alloc_temp(basis.Pz)
    )

function precompute_A!(A, tmp, alist, R1, Z1, Rs, Zs, basis::EnvPairBasis)
   # construct the coordinate system, and convert
   C = CylindricalCoordinateSystem(R1)
   # loop through the environment to assemble the As
   for (R, Z) in zip(Rs, Zs)
      rθz = cylindrical(C, R)
      evaluate!(tmp.Pr, tmp.tmp_Pr, basis.Pr, rθz.r)
      evaluate!(tmp.Pz, tmp.tmp_Pz, basis.Pz, rθz.z)
      evaluate!(tmp.Pθ, tmp.tmp_Pθ, basis.Pθ, rθz)
      iz = z2i(ship, Z)
      for i = alist.firstz[iz]:(alist.firstz[iz+1]-1)
         zklm = alist[i]
         A[i] += tmp.Pr[zklm.k+1] * tmp.Pθ[cyl_l2i(zklm.l)] * tmp.Pz[zklm.m+1]
      end
   end
end


function evaluate!(B::AbstractVector{Complex{T}},
                   tmp,
                   basis::EnvPairBasis{T},
                   R0::SVector{T},
                   Renv::AbstractVector)

   # construct the basis for the r1-variable
   r0 = norm(R0)
   P0 = evaluate!(tmp.P0, tmp.tmp_P0, basis.P0, r0)

   # evaluate the A-basis, i.e. the density projections of the environment
   A = precompute_A!(tmp, basis, Renv) # TODO

   # loop over all basis functions
   for i = 1:length(aalist)
      B[i] = P0[aalist.i2m0[i] + 1]
      for α = 1:aalist.len[i]
         B[i] *= A[aalist.i2Aidx[i, α]]  # TODO: reverse i, α
      end
   end

   return B
end
