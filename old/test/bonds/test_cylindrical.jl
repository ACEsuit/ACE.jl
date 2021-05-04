
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------


@testset "Cylindrical Coordinates" begin

@info("Testset Cylindrical Coordinates")
##

using StaticArrays, Test
using LinearAlgebra
using JuLIP.Testing: print_tf

using ACE
using ACE.Bonds: CylindricalCoordinateSystem, cylindrical,
                  cartesian, CylindricalCoordinates


##
for ntest = 1:7
   R = 1.0 .+ rand(SVector{3, Float64})
   R̂ = R/norm(R)
   o = R/2
   C = CylindricalCoordinateSystem(R, o)
   for mtest = 1:7
      r = rand(SVector{3, Float64}) .- 0.5
      c = cylindrical(C, r)
      r1 = cartesian(C, c)
      print_tf(@test r ≈ r1)
      cref = CylindricalCoordinates(c.cosθ, c.sinθ, c.r, - c.z)
      rref = r - 2 * dot(r-o, R̂) * R̂
      print_tf(@test rref ≈ cartesian(C, cref))
   end
end

##

end
