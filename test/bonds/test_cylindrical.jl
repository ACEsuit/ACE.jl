
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


using StaticArrays, Test
using LinearAlgebra
using JuLIP.Testing: print_tf

using PoSH
using PoSH.Bonds: CylindricalCoordinateSystem, cylindrical,
                  cartesian

for ntest = 1:10
   R = 1.0 .+ rand(SVector{3, Float64})
   C = CylindricalCoordinateSystem(R)
   for mtest = 1:10
      r = rand(SVector{3, Float64}) .- 0.5
      c = cylindrical(C, r)
      r1 = cartesian(C, c)
      print_tf(@test r ≈ r1)
   end
end
