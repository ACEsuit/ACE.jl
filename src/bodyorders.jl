
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------



# -------------------------------------------------------------
#       construct l,k tuples that specify basis functions
# -------------------------------------------------------------

# keep this for the sake of a record and comparison with the general case
function filter_tuples(KL, Nu, ::Val{2}, cg)  # 3B version
   keep = fill(true, length(Nu))
   for (i, ν) in enumerate(Nu)
      kl1, kl2 = KL[ν[1]], KL[ν[2]]
      if kl1.l != kl2.l
         keep[i] = false
      end
   end
   return Nu[keep]
end

# keep this for the sake of a record and comparison with the general case
function filter_tuples(KL, Nu, ::Val{3}, cg)  # 4B version
   keep = fill(true, length(Nu))
   for (i, ν) in enumerate(Nu)
      l1, l2, l3 = KL[ν[1]].l, KL[ν[2]].l, KL[ν[3]].l
      if !( (abs(l1-l2) <= l3 <= l1+l2) && iseven(l1+l2+l3) )
         keep[i] = false
      end
   end
   return Nu[keep]
end

function filter_tuples(KL, Nu, ::Val{4}, cg)
   keep = fill(true, length(Nu))
   for (i, ν) in enumerate(Nu)
      ll = SVector(ntuple(i -> KL[ν[i]].l, 4))
      # invariance under reflections
      if !iseven(sum(ll))
         keep[i] = false
         continue
      end
      # requirement to define the CG coefficients
      if max(abs(ll[1]-ll[2]), abs(ll[3]-ll[4])) > min(ll[1]+ll[2], ll[3]+ll[4])
         keep[i] = false
         continue
      end

      # The next part is purely a health-check, that should not be necessary!
      # basically we are checking that all basis functions that we are
      # retaining are really non-zero!
      foundnz = false
      for mpre in _mrange(ll)
         mm = SVector(Tuple(mpre)..., -sum(Tuple(mpre)))
         if abs(mm[end]) > ll[4]; continue; end
         if _Bcoeff(ll, mm, cg) != 0.0
            foundnz = true
         end
      end
      @assert foundnz
   end
   return Nu[keep]
end
