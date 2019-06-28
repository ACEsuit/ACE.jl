
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


@testset "Fast SHIP Implementation" begin

using SHIPs, JuLIP, BenchmarkTools, LinearAlgebra, Test, Random, StaticArrays
using SHIPs: eval_basis!, eval_basis

function randR()
   R = rand(JVecF) .- 0.5
   return (0.9 + 2 * rand()) * R/norm(R)
end
randR(N) = [ randR() for n=1:N ]
randcoeffs(B) = rand(length(B)) .* (1:length(B)).^(-2)

##

trans3 = PolyTransform(3, 1.0)
B3 = SHIPBasis(3, 13, 2.0, trans3, 2, 0.5, 3.0)
trans2 = PolyTransform(2, 1.3)
B2 = SHIPBasis(2, 15, 2.0, trans2, 2, 0.5, 3.0)
B4 = SHIPBasis(4, 12, 2.0, trans3, 2, 0.5, 3.0)
BB = [B2, B3, B4]

##

@info("--------------- Fast 🚢 Implementation ---------------")

@info("Testing correctness of `SHIP` against `SHIPBasis`")
for B in BB
   @info("   bodyorder = $(SHIPs.bodyorder(B))")
   coeffs = randcoeffs(B)
   ship = SHIP(B, coeffs)
   @info("   test (de-)dictionisation")
   println(@test decode_dict(Dict(ship)) == ship)
   @show length(B), length(ship)
   store = SHIPs.alloc_temp(ship)
   @info("      check that SHIPBasis ≈ SHIP")
   for ntest = 1:30
      Rs = randR(20)
      Es = SHIPs.evaluate!(ship, Rs, store)
      Bs = dot(coeffs, SHIPs.eval_basis(B, Rs))
      print_tf(@test Es ≈ Bs)
   end
   println()
   # @info("      Quick timing test")
   # Rs = randR(50)
   # Btmp = SHIPs.alloc_B(B)
   # print("       SHIPBasis : "); @btime SHIPs.eval_basis!($Btmp, $B, $Rs)
   # print("            SHIP : "); @btime SHIPs.evaluate!($ship, $Rs, $store)
   # println()
end


##
@info("Check Correctness of SHIP gradients")
for B in BB
   @info("   body-order = $(SHIPs.bodyorder(B))")
   coeffs = randcoeffs(B)
   ship = SHIP(B, coeffs)
   Rs = randR(20)
   store = SHIPs.alloc_temp_d(ship, Rs)
   dEs = zeros(JVecF, length(Rs))
   Es = SHIPs.evaluate_d!(dEs, ship, Rs, store)
   Esb = SHIPs.evaluate!(ship, Rs, store)
   @info("      Correctness of Es from evaluate_d!")
   println(@test Es ≈ Esb)
   @info("      Correctness of directional derivatives")
   for ndir = 1:20
      U = [rand(JVecF) .- 0.5 for _=1:length(Rs)]
      errs = Float64[]
      for p = 2:10
         h = 0.1^p
         dEs_U = dot(dEs, U)
         dEs_h = (SHIPs.evaluate!(ship, Rs + h * U, store) - Es) / h
         push!(errs, abs(dEs_h - dEs_U))
      end
      success = (/(extrema(errs)...) < 1e-3) || (minimum(errs) < 1e-10)
      print_tf(@test success)
   end
   println()
end


##
@info("Check Correctness of SHIP calculators")
@warn("    [TODO]")

end
