
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


module TB

import JuLIP
using JuLIP.MLIPs: IPBasis
using JuLIP.Potentials: SZList, ZList

import JuLIP: alloc_temp, alloc_temp_d
import JuLIP.Potentials: evaluate!
import Base: Dict, convert, ==

import SHIPs: SHIPBasis, fcut, precompute_A!, eval_basis!, fcut, transform,
              _my_mul!
using SHIPs.SphericalHarmonics: index_y

struct SHBondBasis{T, NZ, TJ}
   shp::SHIPBasis{T, NZ, TJ}
   fcut::SHBCutoff
end

function SHBondBasis(shpB::SHIPBasis)

end

function evaluate!(B, tmp, shb, R0, Rs)
   # generate a Zs that separates the first particle from the rest
   # and the corresponding Rs
   Zs = tmp.shb_Zs
   Zs[1] = 0
   allRs = tmp.shb_Rs
   allRs[1] = R0
   idx = 1
   for R in Rs
      fc = fcut(R, R0, shb)
      if fc != 0
         idx += 1
         Zs[idx] = shb.shp.zlist[1]
         Rs[idx] = R
      end
   end
   # now we can compute the As and AAs
   precompute_A!(tmp.A[1], tmp, shb.shp.alists[1], Rs, Zs, shb)
   precompute_AA!(tmp.AA[1], tmp.A[1], ship.aalists[1])
   # and finally fill the Bs
   fill!(B, 0)
   _my_mul!(tmp.Bc[1], ship.A2B[1], tmp.AA[1])
   B .= real.(tmp.Bc[1])
   return B
end


function fcut(R, R0, shb::SHBondBasis)
   J = shb.shp.J
   r0, r1 = norm(R), norm(R - R0)
   x0, x1 = corr_transform(J, r0), corr_transform(J, r1)
   # ----------------------------------------------------------------------
   return fcut(J, r0, x0) + fcut(J, r1, x1)
end

function precompute_A!(A, tmp, alist, Rs, Zs, shb::SHBondBasis)
   fill!(A, 0)
   for (R, Z) in zip(Rs, Zs)
      # evaluate the r-basis and the R̂-basis for the current neighbour at R
      # first evaluate the cut-off again
      fc = fcut(R, Rs[1], shb)
      r = norm(R)
      x = corr_transform(shb.shp.J, r)
      # then the radial basis - but by passing x, fc we only evaluate the
      # Jacobi polynomials *without* the cutoff component!
      eval_basis!(tmp.J, tmp.tmpJ, ship.J, r, x, fc)
      # ... then the SHs
      eval_basis!(tmp.Y, tmp.tmpY, ship.SH, R)
      # add the contributions to the A_zklm
      iz = z2i(shb.shp, Z)
      for i = alist.firstz[iz]:(alist.firstz[iz+1]-1)
         zklm = alist[i]
         A[i] += tmp.J[zklm.k+1] * tmp.Y[index_y(zklm.l, zklm.m)]
      end
   end
   return A
end



end
