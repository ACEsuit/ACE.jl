
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


@testset "Fourier  Basis" begin

@info("Testset FourierBasis")

##

using StaticArrays, Test
using LinearAlgebra
using JuLIP.Testing: print_tf

using SHIPs
using JuLIP: evaluate!, evaluate, evaluate_d!
using SHIPs: alloc_B, alloc_dB
using SHIPs.Bonds: CylindricalCoordinateSystem, cylindrical,
                  cartesian,
                  FourierBasis,
                  CylindricalCoordinates


##

deg = 10
fB = FourierBasis(10, Float64)

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
   Px = [ exp(im * l * θ) for l = -deg:deg ]
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
