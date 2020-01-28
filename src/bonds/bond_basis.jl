
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


# k : r-degree
# l : θ-degree
# m : z-degree

import PoSH: evaluate
using PoSH: AList

struct EnvPairBasis
   Pr::TR               # specifies the radial basis
   Pz::Tz               # specifies the z-basis
   Pθ::Tθ               # the angular basis
   alist::AList

end


function precompute_A!(A, tmp, alist, R1, Z1, Rs, Zs, basis::EnvPairBasis)
   # construct the coordinate system, and convert
   C = CylindricalCoordinateSystem(R1)
   # loop through the envinroment to assemble the As 
   for (R, Z) in zip(Rs, Zs)
      rθz = cylindrical(C, R)
      evaluate!(tmp.Pr, tmp.tmp_Pr, basis.Pr, rθz.r)
      evaluate!(tmp.Pz, tmp.tmp_Pz, basis.Pz, rθz.z)
      evaluate!(tmp.Pθ, tmp.tmp_Pθ, basis.Pθ, rθz)
      iz = z2i(ship, Z)
      for i = alist.firstz[iz]:(alist.firstz[iz+1]-1)
         zklm = alist[i]
         A[i] += tmp.Pr[zklm.k+1] * tmp.Pθ[_l2i(zklm.l)] * tmp.Pz[zklm.m+1]
      end
   end
end


function evaluate!(B::AbstractVector{T},
                   tmp,
                   basis::EnvPairBasis{T},
                   R1::SVector{T},
                   Renv::AbstractVector)

   # construct the basis for the r1-variable
   r1 = norm(R1)

   # evaluate the A-basis, i.e. the density projections
   precompute_Acyl!(tmp, basis, cylenv)

   #




end
