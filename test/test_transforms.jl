
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


# - for both 1s and 2s and p = 2, 3, 4
# - consistency of derivatives (finite-difference test)
# - L2 orthogonality

using SHIPs, Printf, Test, LinearAlgebra
using SHIPs: PolyTransform, rbasis, eval_basis, eval_basis_d
using SHIPs.JacobiPolys: Jacobi



verbose = false

for p in 2:4
   @info("p = $p, random transform")
   trans = PolyTransform(1+rand(), 1+rand())
   B1 = rbasis(10, trans, 2, 3.0)
   B2 = rbasis(10, trans, 2, 0.5, 3.0)
   for B in [B1, B2]
      B == B1 && @info("basis = 1s")
      B == B2 && @info("basis = 2s")
      for r in [3 * rand(10); [3.0]]
         P, dP = eval_basis_d(B, r)
         errs = []
         verbose && @printf("     h    |     error  \n")
         for p = 2:10
            h = 0.1^p
            dPh = (eval_basis(B, r+h) - P) / h
            push!(errs, norm(dPh - dP, Inf))
            verbose && @printf(" %.2e | %2e \n", h, errs[end])
         end
         print(@test (/(extrema(errs)...) < 1e-3) || (minimum(errs) < 1e-10) )
      end
      println()
   end
end
