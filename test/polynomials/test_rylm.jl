
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------


@testset "Real  Ylm" begin

##
import ACE
using JuLIP.Testing
using LinearAlgebra, StaticArrays, BenchmarkTools, Test, Printf
using ACE.SphericalHarmonics
using ACE.SphericalHarmonics: dspher_to_dcart, PseudoSpherical,
               cart2spher, spher2cart, RSHBasis, index_y
using JuLIP: evaluate, evaluate_d, evaluate_ed

verbose = false

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
   R = ACE.Random.rand_sphere()
   cY = evaluate(cSH, R)
   rY = evaluate(rSH, R)
   print_tf(@test test_r2c(maxL, cY, rY))
end
println()

##

@info("Test: check derivatives of real spherical harmonics")
for nsamples = 1:30
   R = @SVector rand(3)
   SH = RSHBasis(5)
   Y, dY = evaluate_ed(SH, R)
   DY = Matrix(transpose(hcat(dY...)))
   errs = []
   verbose && @printf("     h    | error \n")
   for p = 2:10
      h = 0.1^p
      DYh = similar(DY)
      Rh = Vector(R)
      for i = 1:3
         Rh[i] += h
         DYh[:, i] = (evaluate(SH, SVector(Rh...)) - Y) / h
         Rh[i] -= h
      end
      push!(errs, norm(DY - DYh, Inf))
      verbose && @printf(" %.2e | %.2e \n", h, errs[end])
   end
   success = (minimum(errs[2:end]) < 1e-3 * maximum(errs[1:3])) || (minimum(errs) < 1e-10)
   print_tf(@test success)
end
println()


end
