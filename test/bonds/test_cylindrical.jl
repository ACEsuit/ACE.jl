
@testset "Cylindrical Coordinates" begin

@info("Testset Cylindrical Coordinates") 
##

using StaticArrays, Test
using LinearAlgebra
using JuLIP.Testing: print_tf

using SHIPs
using SHIPs.Bonds: CylindricalCoordinateSystem, cylindrical,
                  cartesian

##

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


end
