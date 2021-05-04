
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------


@testset "Fourier  Basis" begin

@info("Testset FourierBasis")

##

using StaticArrays, Test
using LinearAlgebra
using JuLIP.Testing: print_tf

using ACE
using JuLIP: evaluate!, evaluate, evaluate_d!
using ACE: alloc_B, alloc_dB
using ACE.Bonds: CylindricalCoordinateSystem, cylindrical,
                  cartesian,
                  FourierBasis,
                  CylindricalCoordinates


##

deg = 10
fB = FourierBasis(10, Float64)

##

@info("test cyl_l2i ∘ cyl_i2l = id")
is = sort([ ACE.Bonds.cyl_l2i(k) for k = -deg:deg ])
println(@test is == 1:(2*deg+1))
for i = 1:(2*deg+1)
   k = ACE.Bonds.cyl_i2l(i)
   print_tf(@test ACE.Bonds.cyl_l2i(k) == i)
end

##
# choose a few random angles, then confirm that the fourier basis
# does the right thing against an explicit expression using a
# comprehension

@info("Test correctness of Fourier / Trigonometric Polynomial basis")
for ntest = 1:30
   θ = 2 * π * rand()
   cosθ, sinθ = cos(θ), sin(θ)
   c = CylindricalCoordinates(cosθ, sinθ, 0.0, 0.0)

   P = evaluate!(alloc_B(fB), nothing, fB, c)
   Px = zeros(ComplexF64, 2*deg+1)
   for l = -deg:deg
      Px[ACE.Bonds.cyl_l2i(l)] = exp(im * l * θ)
   end
   print_tf(@test P ≈ Px)
end

##

@info("Test correctness of gradient of Fourier Basis")
P = alloc_B(fB)
Ph = alloc_B(fB)
dP = alloc_dB(fB)

for ntest = 1:30
   θ = 2 * π * rand()
   cosθ, sinθ = cos(θ), sin(θ)
   c = CylindricalCoordinates(cosθ, sinθ, 0.0, 0.0)

   evaluate_d!(P, dP, nothing, fB, c)
   P1 = evaluate!(alloc_B(fB), nothing, fB, c)
   print_tf(@test P1 ≈ P)
   dP_dθ = [ dp[1] * (-sinθ) + dp[2] * cosθ  for dp in  dP ]

   errs = Float64[]
   for p = 2:10
      h = 0.1^p
      θh = θ + h
      ch = CylindricalCoordinates(cos(θh), sin(θh), 0.0, 0.0)
      evaluate!(Ph, nothing, fB, ch)
      dhP_dθ = (Ph - P) / h
      push!(errs, norm(dP_dθ - dhP_dθ, Inf))
   end
   print_tf(@test minimum(errs[2:end]) < 1e-3 * maximum(errs[1:3]))
end

##

end
