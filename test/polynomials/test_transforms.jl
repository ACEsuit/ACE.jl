
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------



@testset "Transforms" begin

using ACE, Printf, Test, LinearAlgebra, JuLIP, JuLIP.Testing
using JuLIP: evaluate, evaluate_d

verbose = false
maxdeg = 10

@info("Testing Transforms and TransformedPolys")
for p in 2:4
   @info("p = $p, random transform")
   trans = PolyTransform(1+rand(), 1+rand())
   @info("      test (de-)dictionisation")
   println(@test read_dict(write_dict(trans)) == trans)
   B1 = transformed_jacobi(maxdeg, trans, 3.0; pcut = p)
   B2 = transformed_jacobi(maxdeg, trans, 3.0, 0.5, pin = p, pcut = p)
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

#---
@info("Testing PolyTransforms")
for p = 2:4
   r0 = 1+rand()
   trans = PolyTransform(p, r0)
   ACE.Testing.test_transform(trans, [r0/2, 3*r0])
end

#---
@info("Testing Morse Transform")
for lam = 1.0:3.0
   r0 = 1+rand()
   trans = ACE.Transforms.MorseTransform(lam, r0)
   ACE.Testing.test_transform(trans, [r0/2, 3*r0])
end

#---

@info("Testing Agnesi Transform")
for p = 2:4
   r0 = 1+rand()
   trans = ACE.Transforms.AgnesiTransform(r0, p)
   ACE.Testing.test_transform(trans, [r0/2, 3*r0])
end

#---

# using Plots
# r0 = 1.0
# rr = range(0.0, 3*r0, length=200)
# plot(; size = (500, 300))
# for p = 2:4
#    tpoly = PolyTransform(p, r0)
#    tagnesi = ACE.Transforms.AgnesiTransform(r0, p)
#    plot!(rr, tagnesi.(rr), lw=2, c=p-1, label = "p = $p")
#    plot!(rr, tpoly.(rr), lw=2, c=p-1, ls = :dash, label = "")
# end
# xlabel!("r")
# ylabel!("x")
# title!("solid = Agnesi, dashed = Poly")
# vline!([1.0], lw=2, c=:black, label = "r0")
# ylims!(0.0, 2.0)
#---
end
