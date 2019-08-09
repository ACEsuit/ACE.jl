
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------



using SHIPs, SHIPs.SphericalHarmonics, StaticArrays, LinearAlgebra
using SHIPs: _mrange


function compute_Ckm(ll::SVector{4})
   cg = ClebschGordan(sum(ll))

   len = 0
   for mm in _mrange(ll)
      len += 1
   end

   Ckm = zeros(len, len)

   for (im, mm) in enumerate(_mrange(ll)), (ik, kk) in enumerate(_mrange(ll))
      jlo = max(abs(ll[1]-ll[2]), abs(ll[3]-ll[4]))
      jhi = min(ll[1]+ll[2], ll[3]+ll[4])
      for j = jlo:jhi
         if (abs(mm[1]+mm[2]) > j) || (abs(mm[3]+mm[4]) > j) ||
            (abs(kk[1]+kk[2]) > j) || (abs(kk[3]+kk[4]) > j)
            continue
         end
         Ckm[ik,im] += (-1)^(mm[1]+mm[2]-kk[1]-kk[2]) / (2*j+1) *
                       cg(ll[1], mm[1], ll[2], mm[2], j, mm[1]+mm[2]) *
                       cg(ll[3], mm[3], ll[4], mm[4], j, mm[3]+mm[4]) *
                       cg(ll[1], kk[1], ll[2], kk[2], j, kk[1]+kk[2]) *
                       cg(ll[3], kk[3], ll[4], kk[4], j, kk[3]+kk[4])
      end
   end
   return Ckm
end


# CASE 1
ll1 = SVector(2,1,1,2)
Ckm = compute_Ckm(ll1)
@show rank(Ckm)
svdf = svd(Ckm)
@show svdf.S[1:5]
for i = 1:3
   @info("V[:,$i]")
   @show round.(svdf.Vt[i,:], digits=2)
end

# CASE 2
ll2 = SVector(2,3,4,3)
Ckm = compute_Ckm(ll2)
@show rank(Ckm)
@show svdvals(Ckm)[1:8]

# CASE 3
ll3 = SVector(5,4,4,3)
Ckm = compute_Ckm(ll3)
@show rank(Ckm)
@show svdvals(Ckm)[1:10]
