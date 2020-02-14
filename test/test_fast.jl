
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


@testset "Fast SHIP Implementation" begin

##
using PoSH, JuLIP, BenchmarkTools, LinearAlgebra, Test, Random, StaticArrays
using JuLIP
using JuLIP: evaluate, evaluate_d
using JuLIP.Testing

function randR()
   R = rand(JVecF) .- 0.5
   return (0.9 + 2 * rand()) * R/norm(R)
end
randR(N) = [ randR() for n=1:N ], zeros(Int16, N), 0
randcoeffs(B) = rand(length(B)) .* (1:length(B)).^(-2)

##

trans = PolyTransform(3, 1.0)
fcut = PolyCutoff2s(2, 0.5, 3.0)
B2 = SHIPBasis(SparseSHIP(2, 10, wL=2.0), trans, fcut)
B3 = SHIPBasis(SparseSHIP(3,  8, wL=2.0), trans, fcut)
B4 = SHIPBasis(SparseSHIP(4,  7, wL=2.0), trans, fcut)
B5 = SHIPBasis(SparseSHIP(5,  6, wL=2.0), trans, fcut)
BB = [B2, B3, B4, B5]

##


@info("--------------- Fast 🚢 Implementation ---------------")

@info("Testing correctness of `SHIP` against `SHIPBasis`")
for B in BB
   @info("   bodyorder = $(PoSH.bodyorder(B))")
   coeffs = randcoeffs(B)
   ship = SHIP(B, coeffs)
   @info("   test (de-)dictionisation")
   println(@test decode_dict(Dict(ship)) == ship)
   @show length(B), length(ship)
   tmp = PoSH.alloc_temp(ship, 10)
   @info("      check that SHIPBasis ≈ SHIP")
   for ntest = 1:30
      Rs, Zs, z0 = randR(10)
      Es = evaluate!(tmp, ship, Rs, Zs, z0)
      Bs = dot(coeffs, evaluate(B, Rs, Zs, z0))
      print_tf(@test Es ≈ Bs)
   end
   println()
   # ------------------------------------------------------------
   # @info("      Quick timing test")
   # Nr = 30
   # Rs, Zs, z0 = randR(Nr)
   # b = PoSH.alloc_B(B)
   # tmp = PoSH.alloc_temp(ship, Nr)
   # tmpB = PoSH.alloc_temp(B, Nr)
   # print("       SHIPBasis : "); @btime evaluate!($b, $tmpB, $B, $Rs, $Zs, $z0)
   # print("            SHIP : "); @btime evaluate!($tmp, $ship, $Rs, $Zs, $z0)
   # println()
end


##
@info("Check Correctness of SHIP gradients")
for B in BB
   @info("   body-order = $(PoSH.bodyorder(B))")
   coeffs = randcoeffs(B)
   ship = SHIP(B, coeffs)
   Rs, Zs, z0 = randR(10)
   tmp = PoSH.alloc_temp_d(ship, length(Rs))
   dEs = zeros(JVecF, length(Rs))
   evaluate_d!(dEs, tmp, ship, Rs, Zs, z0)
   Es = evaluate!(tmp, ship, Rs, Zs, z0)
   println(@test Es ≈ evaluate(ship, Rs, Zs, z0))
   println(@test dEs ≈ evaluate_d(ship, Rs, Zs, z0))
   @info("      Correctness of directional derivatives")
   for ndir = 1:20
      U = [rand(JVecF) .- 0.5 for _=1:length(Rs)]
      errs = Float64[]
      for p = 2:10
         h = 0.1^p
         dEs_U = dot(dEs, U)
         dEs_h = (evaluate!(tmp, ship, Rs + h * U, Zs, z0) - Es) / h
         push!(errs, abs(dEs_h - dEs_U))
      end
      success = (/(extrema(errs)...) < 1e-3) || (minimum(errs) < 1e-10)
      print_tf(@test success)
   end
   println()
end


##
@info("Check Correctness of SHIP calculators")

naive_energy(ship::SHIP, at) =
      sum( evaluate(ship, R, at.Z[j], at.Z[i])
            for (i, j, R) in sites(at, cutoff(ship)) )

for B in BB
   @info("   body-order = $(PoSH.bodyorder(B))")
   coeffs = randcoeffs(B)
   ship = SHIP(B, coeffs)
   at = bulk(:Si) * 3
   at.Z[:] .= 0
   rattle!(at, 0.1)
   print("     energy: ")
   println(@test energy(ship, at) ≈ naive_energy(ship, at) )
   print("site-energy: ")
   println(@test energy(ship, at) ≈ sum( site_energy(ship, at, n)
                                         for n = 1:length(at) ) )
   println("forces: ")
   println(@test JuLIP.Testing.fdtest(ship, at))
   println("site-forces: ")
   println(@test JuLIP.Testing.fdtest( x -> site_energy(ship, set_dofs!(at, x), 3),
                                       x -> mat(site_energy_d(ship, set_dofs!(at, x), 3))[:],
                                       dofs(at) ) )
end

##

@info("Test JSON (de-)serialisation of PoSH")
coeffs = randcoeffs(B5)
ship = SHIP(B5, coeffs)
println(@test decode_dict(Dict(ship)) == ship)
save_json("tmp.json", Dict("IP" => Dict(ship)))
IP = decode_dict( load_json("tmp.json")["IP"])
println(@test IP == ship)
run(`rm tmp.json`)

##
end
