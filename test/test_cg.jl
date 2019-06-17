
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


@testset "Clebsch-Gordan" begin

using PyCall, Test, SHIPs.SphericalHarmonics
using SHIPs: eval_basis
using SHIPs.SphericalHarmonics: index_y

sympy = pyimport("sympy")
spin = pyimport("sympy.physics.quantum.spin")


pycg(j1, m1, j2, m2, j3, m3, T=Float64) =
      spin.CG(j1, m1, j2, m2, j3, m3).doit().evalf().__float__()

@info("Testing cg1 implementation against sympy ... ")
for j1 = 0:2, j2=0:2, j3=0:4
   for m1 = -j1:j1, m2=-j2:j2, m3=-j3:j3
      @test cg1(j1,m1,j2,m2,j3,m3) ≈ pycg(j1,m1, j2,m2, j3,m3)
   end
end

@info("Check CG Coefficients for some higher quantum numbers...")
j1 = 8
j2 = 11
j3 = j1+j2
for m1 = -j1:j1, m2=-j2:j2, m3=-j3:j3
   @test cg1(j1,m1,j2,m2,j3,m3) ≈ pycg(j1,m1, j2,m2, j3,m3)
end


@info("Checking the SphH expansion in terms of CG coeffs")
# expansion coefficients of a product of two spherical harmonics in terms a
# single spherical harmonic
# see e.g. https://en.wikipedia.org/wiki/Clebsch–Gordan_coefficients
# this is the magic formula that we need
for ntest = 1:10
   # two random Ylm  ...
   l1, l2 = rand(1:10), rand(1:10)
   m1, m2 = rand(-l1:l1), rand(-l2:l2)
   # ... evaluated at random spherical coordinates
   θ = rand() * π
   φ = (rand()-0.5) * 2*π
   R = SVector( cos(φ)*sin(θ), sin(φ)*sin(θ), cos(θ) )
   # evaluate all relevant Ylms (up to l1 + l2)
   Ylm = eval_basis(SHBasis(l1+l2), R)
   # evaluate the product p = Y_l1_m1 * Y_l2_m2
   p = Ylm[index_y(l1,  m1)] * Ylm[index_y(l2,m2)]
   # and its expansion in terms of CG coeffs
   p2 = 0.0
   M = m1 + m2  # all other coeffs are zero

   for L = abs(M):(l1+l2)
      p2 += sqrt( (2*l1+1)*(2*l2+1) / (4 * π * (2*L+1)) ) *
            cg1(l1,  0, l2,  0, L, 0) *
            cg1(l1, m1, l2, m2, L, M) *
            Ylm[index_y(L, M)]
   end
   print(@test p ≈ p2); print(" ")
end
println()

end # @testset
