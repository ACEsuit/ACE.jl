
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


@testset "Clebsch-Gordan" begin

@info("Testset Clebsch-Gordan")

##

using PyCall, Test, SHIPs, SHIPs.SphericalHarmonics, JuLIP.Testing, StaticArrays
using JuLIP: evaluate
using SHIPs.SphericalHarmonics: index_y
using SHIPs.Rotations3D: ClebschGordan

##

sympy = pyimport("sympy")
spin = pyimport("sympy.physics.quantum.spin")

pycg(j1, m1, j2, m2, j3, m3, T=Float64) =
      spin.CG(j1, m1, j2, m2, j3, m3).doit().evalf().__float__()

cg = SHIPs.Rotations3D.ClebschGordan()

j1 = 3
j2 = 4
m1 = -3
m2 = 2
J = 4
M = m1 + m2
cg(j1, j2, m1, m2, J, M)

##

@info("compare implementation against `sympy`")
ntest = 0
while ntest <= 200
   j1 = rand(0:10)
   j2 = rand(0:10)
   J = rand(abs(j1-j2):min(10,(j1+j2)))
   m1 = rand(-j1:j1)
   for m2 = -j2:j2
      M = m1+m2
      if abs(M) <= J
         global ntest += 1
         print_tf(@test cg(j1,m1, j2,m2, J,M) ≈ pycg(j1,m1, j2,m2, J,M))
         # print_tf(@test clebschgordan(j1,m1, j2,m2, J,M) ≈ )
      end
   end
end
println()

##

@info("Checking the SphH expansion in terms of CG coeffs")
# expansion coefficients of a product of two spherical harmonics in terms a
# single spherical harmonic
# see e.g. https://en.wikipedia.org/wiki/Clebsch–Gordan_coefficients
# this is the magic formula that we need, on which everything else is based 
for ntest = 1:200
   # two random Ylm  ...
   l1, l2 = rand(1:10), rand(1:10)
   m1, m2 = rand(-l1:l1), rand(-l2:l2)
   # ... evaluated at random spherical coordinates
   θ = rand() * π
   φ = (rand()-0.5) * 2*π
   R = SVector( cos(φ)*sin(θ), sin(φ)*sin(θ), cos(θ) )
   # evaluate all relevant Ylms (up to l1 + l2)
   Ylm = evaluate(SHBasis(l1+l2), R)
   # evaluate the product p = Y_l1_m1 * Y_l2_m2
   p = Ylm[index_y(l1,  m1)] * Ylm[index_y(l2,m2)]
   # and its expansion in terms of CG coeffs
   p2 = 0.0
   M = m1 + m2  # all other coeffs are zero

   for L = abs(M):(l1+l2)
      p2 += sqrt( (2*l1+1)*(2*l2+1) / (4 * π * (2*L+1)) ) *
            cg(l1,  0, l2,  0, L, 0) *
            cg(l1, m1, l2, m2, L, M) *
            Ylm[index_y(L, M)]
   end
   print_tf((@test (p ≈ p2) || (abs(p-p2) < 1e-15)))
end
println()

##

end # @testset
