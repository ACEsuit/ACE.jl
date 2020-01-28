
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


@testset "Real  Ylm" begin

##
import PoSH
using JuLIP.Testing
using LinearAlgebra, StaticArrays, BenchmarkTools, Test, Printf
using PoSH.SphericalHarmonics
using PoSH.SphericalHarmonics: dspher_to_dcart, PseudoSpherical,
               cart2spher, spher2cart, RSHBasis, index_y
using JuLIP: evaluate, evaluate_d, evaluate_ed

verbose = true

##
@info("Testing consistency of Real and Complex SH; Condon-Shortley convention")

function test_r2c(L, cY, rY)
   cYt = similar(cY)
   for l = 0:L
      m = 0
      i = index_y(l, m)
      cYt[i] = rY[i]
      for m = 1:l
         i_p = index_y(l, m)
         i_m = index_y(l, -m)
         # test the expressions
         #  Y_l^m    =      1/√2 (Y_{lm} - i Y_{l,-m})
         #  Y_l^{-m} = (-1)^m/√2 (Y_{lm} + i Y_{l,-m})
         cYt[i_p] = (1/sqrt(2)) * (rY[i_p] - im * rY[i_m])
         cYt[i_m] = (-1)^m * (1/sqrt(2)) * (rY[i_p] + im * rY[i_m])
      end
   end
   return cY ≈ cYt
end

maxL = 20
cSH = SHBasis(maxL)
rSH = RSHBasis(maxL)

for nsamples = 1:30
   R = PoSH.Utils.rand_sphere()
   cY = evaluate(cSH, R)
   rY = evaluate(rSH, R)
   print_tf(@test test_r2c(maxL, cY, rY))
end
println()


end
