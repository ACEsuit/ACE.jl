
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------



@testset "Transforms" begin

using PoSH, Printf, Test, LinearAlgebra
using PoSH: PolyTransform, TransformedJacobi
using PoSH.JacobiPolys: Jacobi
using PoSH.SphericalHarmonics
using PoSH.SphericalHarmonics: dspher_to_dcart, PseudoSpherical,
               cart2spher, spher2cart

using JuLIP: evaluate, evaluate_d


verbose = false

for p in 2:4
   @info("p = $p, random transform")
   trans = PolyTransform(1+rand(), 1+rand())
   @info("      test (de-)dictionisation")
   println(@test decode_dict(Dict(trans)) == trans)
   B1 = TransformedJacobi(10, trans, PolyCutoff1s(2, 3.0))
   B2 = TransformedJacobi(10, trans, PolyCutoff2s(2, 0.5, 3.0))
   for B in [B1, B2]
      B == B1 && @info("basis = 1s")
      B == B2 && @info("basis = 2s")
      for r in [3 * rand(10); [3.0]]
         P = evaluate(B, r)
         dP = evaluate_d(B, r)
         errs = []
         verbose && @printf("     h    |     error  \n")
         for p = 2:10
            h = 0.1^p
            dPh = (evaluate(B, r+h) - P) / h
            push!(errs, norm(dPh - dP, Inf))
            verbose && @printf(" %.2e | %2e \n", h, errs[end])
         end
         print_tf(@test (/(extrema(errs)...) < 1e-3) || (minimum(errs) < 1e-10) )
      end
      println()
   end
end


end
