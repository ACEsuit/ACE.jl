

module Orth

include("extimports.jl")

include("aceimports.jl")


# ------------ Products of Spherical Harmonics

import ACE.RPI.Rotations3D: ClebschGordan, cg_conditions
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
   M::Int
   cg::ClebschGordan{T}
end

(coeffs::SHProdCoeffs)(l1, m1, l2, m2) =
      SHProdFixlimi(l1, m1, l2, m2, m1+m2, coeffs.cg)

function Base.getindex(P::SHProdFixlimi{T}, L, M) where {T}
   if cg_conditions(P.l1, P.m1, P.l2, P.m2, L, M) &&
         iseven(P.l1 + P.l2 - L)   # condition on <l1 0 l2 0 | L 0>
      return ( sqrt( (2*P.l1+1)*(2*P.l2+1) / (4*pi*(2*L+1)) )
                 * P.cg(P.l1, P.m1, P.l2, P.m2, L, M)
                 * P.cg(P.l1, 0, P.l2, 0, L, 0) )
   else
      return zero(T)
   end
end


function Base.iterate(P::SHProdFixlimi)
   if !( (abs(P.m1) <= P.l1) && (abs(P.m2) <= P.l2) )  # all CG coeffs are zero
      return nothing
   end
   M = P.M
   L = max( abs(P.l1 - P.l2), abs(M) )  # L ≧ |l1-l2|
   return ( (L, M, P[L, M]), L )
end

function Base.iterate(P::SHProdFixlimi, state)
   L = state; M = P.M   # m1+m2=M is built-in
   L += 1
   if isodd(P.l1 + P.l2 - L)  # skip this iterate if <l1 0 l2 0 | L 0 > = 0
      return iterate(P, L)
   end
   if L > P.l1 + P.l2     # require L ≦ l1+l2, otherwise terminate
      return nothing
   end
   return ( (L, M, P[L, M]), L )
end



# ------------ Products of R_n Y_l^m  basis functions

import ACE.OrthPolys: OrthPolyProdCoeffs, EndlessVector

struct OnepProductCoeffs{T}
   shcoeffs::SHProdFixlimi{T}
   rcoeffs::OrthPolyProdCoeffs{T}
end

struct OnepProdCoeffMatrix{T}
   Psh::SHProdFixlimi{T}
   Pr::EndlessVector{T}
end

(coeffs::OnepProductCoeffs)(n1, l1, m1, n2, l2, m2) =
      OnepProdCoeffMatrix( coeffs.shcoeffs(l1, m1, l2, m2),
                           coeffs.rcoeffs(n1, n1) )

(P::OnepProdCoeffMatrix)(N, L, M) = P.Pr[N] * P.Psh[L, M]


# function Base.iterate(P::OnepProdCoeffMatrix)
#    L = abs(P.l1 - P.l2)
#    M = - L
#    return ( (L, M, P[L, M]), (L, M) )
# end
#
#
# function Base.iterate(P::SHProdFixlimi, state)
# end



end
