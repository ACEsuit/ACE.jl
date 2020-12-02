

module Orth

include("extimports.jl")

include("aceimports.jl")


# ------------ Products of Spherical Harmonics

import ACE.RPI.Rotations3D: ClebschGordan
import ACE.SphericalHarmonics: SHBasis

ClebschGordan

struct SHProdCoeffs{T}
   cg::ClebschGordan{T}
end

SHProdCoeffs(T = Float64) = SHProdCoeffs(ClebschGordan(T))

struct SHProdFixlimi{T}
   l1::Int
   m1::Int
   l2::Int
   m2::Int
   cg::ClebschGordan{T}
end

(coeffs::SHProdCoeffs)(l1, m1, l2, m2) = SHProdFixlimi(l1, m1, l2, m2, coeffs.cg)

function Base.getindex(P::SHProdFixlimi{T}, L, M) where {T}
   if abs(P.l1 - P.l2) <= L <= P.l1 + P.l2
      return ( sqrt( (2*P.l1+1)*(2*P.l2+1) / (4*pi*(2*L+1)) )
                 * P.cg(P.l1, P.m1, P.l2, P.m2, L, M)
                 * P.cg(P.l1, 0, P.l2, 0, L, 0) )
   else
      return zero(T)
   end
end




function Base.iterate(P::SHProdFixlimi)
   L = abs(P.l1 - P.l2)
   M = - L
   return ( (L, M, P[L, M]), (L, M) )
end

function Base.iterate(P::SHProdFixlimi, state)
   L, M = state
   if M == L
      L += 1
      if L > P.l1 + P.l2
         return nothing
      end
      M = -L
   else
      M += 1
   end
   p = P[L, M]
   if abs(p) < 1e-12
      return iterate(P, (L, M))
   end
   return ( (L, M, p), (L, M) )
end




# ------------ Products of R_n Y_l^m  basis functions



end
