
module Regularisers

using PoSH: SHIPBasis, nspecies, _get_I_iz0
using LinearAlgebra: Diagonal

function _get_ww(ship::SHIPBasis{T}, iz0, weightfcn) where {T}
   ww = zeros(T, size(ship.A2B[iz0], 1))
   for ib = 1:length(ship.bgrps[iz0])
      izz, kk, ll =  ship.bgrps[iz0][i]
      Igrp = (ship.firstb[iz0][i]+1):ship.firstb[iz0][i+1]
      ww[Igrp] = weightfcn(kk, ll)
   end
   return ww
end

function _get_ww(ship::SHIPBasis{T}, weightfcn) where {T}
   ww = zeros(length(ship))
   for iz0 = 1:nspecies(ship)
      Iz = _get_I_iz0(ship, iz0)
      ww[Iz] = _get_ww(ship, iz0, weightfcn)
   end
   return ww
end

diffweight(kk, ll, diff) = sqrt(sum(kk.^(2*diff)) + sum(ll.^(2*diff)))

function diagonal_regulariser(shp::SHIPBasis;
                              diff = 0,
                              weightfcn = (kk, ll) -> diffweight(kk, ll, diff))
   return Diagonal(_get_ww(shp, weightfcn))
end

end
