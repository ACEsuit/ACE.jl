
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


@testset "SHIP Multi-Species" begin

##


@info("-------- TEST 🚢  Multi-Species-Basis ---------")
using PoSH, JuLIP, BenchmarkTools, LinearAlgebra, Test, Random, StaticArrays
using PoSH:  PolyCutoff1s, PolyCutoff2s
using JuLIP.MLIPs: IPSuperBasis
using JuLIP.Testing: print_tf
using JuLIP: evaluate!, evaluate, evaluate_d

function randR()
   R = rand(JVecF) .- 0.5
   return (0.9 + 2 * rand()) * R/norm(R)
end

randR(N) = [ randR() for n=1:N ], zeros(Int16, N)

randR(N, syms::NTuple{M, Symbol}) where {M} =
      randR(N, Int16.(atomic_number.(syms)))

randR(N, Zs::NTuple{M, <:Integer}) where {M} =
      randR(N, Int16.(Zs))

randR(N, Zs::NTuple{M, Int16}) where {M}  =
      randR(N)[1], rand(Zs, N), rand(Zs)

function randiso()
   K = @SMatrix rand(3,3)
   K = K - K'
   Q = rand([-1,1]) * exp(K)
end

function randiso(Rs, Zs)
   Q = randiso()
   p = shuffle(1:length(Rs))
   return [ Q * R for  R in Rs[p]], Zs[p]
end

##
@info("Testing creation and (de-)dictionisation of a few BasisSpecs")

spec = SparseSHIP(:X, 3, 10)
println(@test spec == SparseSHIP(3, 10))
println(@test decode_dict(Dict(spec)) == spec)

spec = SparseSHIP([:H, :He], 3, 6)
println(@test spec == SparseSHIP([1,2], 3, 6))
println(@test decode_dict(Dict(spec)) == spec)

spec = SparseSHIP([:H, :He, :Li], 2, 6)
println(@test spec == SparseSHIP([1,2,3], 2, 6))
println(@test decode_dict(Dict(spec)) == spec)

##

trans = PolyTransform(2, 1.0)
cutf = PolyCutoff2s(2, 0.5, 3.0)

ship2 = SHIPBasis(SparseSHIP([1,2], 2, 10, wL=2.0), trans, cutf)
ship3 = SHIPBasis(SparseSHIP([1,2], 3,  8, wL=2.0), trans, cutf)
ship4 = SHIPBasis(SparseSHIP([1,2], 4,  6, wL=1.5), trans, cutf)
ship5 = SHIPBasis(SparseSHIP([1,2], 5,  5, wL=1.5), trans, cutf)
ships = [ship2, ship3, ship4, ship5]

##

@info("Test (de-)dictionisation of basis sets")
for ship in ships
   println(@test (decode_dict(Dict(ship)) == ship))
end


Rs, Zs, iz = randR(20, (1,2))
🚢 = ship3
evaluate(🚢, Rs, Zs, iz)

@info("Test isometry invariance for 3B-6B 🚢 s")
for ntest = 1:20
   Rs, Zs, iz = randR(20, (1,2))
   BB = [ evaluate(🚢, Rs, Zs, iz) for 🚢 in ships ]
   RsX, ZsX = randiso(Rs, Zs)
   BBX = [ evaluate(🚢, RsX, ZsX, iz) for 🚢 in ships ]
   for (B, BX) in zip(BB, BBX)
      print_tf(@test B ≈ BX)
   end
end
println()


##

@info("Test gradients for 3-6B 🚢-basis")
for 🚢 in ships
   @info("  body-order = $(PoSH.bodyorder(🚢)):")
   Rs, Zs, z = randR(20, (1,2))
   tmp = PoSH.alloc_temp_d(🚢, Rs)
   # PoSH.precompute_dA!(tmp, 🚢, Rs, Zs)
   B = evaluate(🚢, Rs, Zs, z)
   dB = PoSH.alloc_dB(🚢, Rs)
   evaluate_d!(dB, tmp, 🚢, Rs, Zs, z)
   @info("      finite-difference test into random directions")
   for ndirections = 1:20
      Us, _ = randR(length(Rs))
      errs = Float64[]
      for p = 2:10
         h = 0.1^p
         Bh = evaluate(🚢, Rs+h*Us, Zs, z)
         dBh = (Bh - B) / h
         dBxU = sum( dot.(Ref(Us[n]), dB[n,:])  for n = 1:length(Rs) )
         push!(errs, norm(dBh - dBxU, Inf))
      end
      success = (/(extrema(errs)...) < 1e-3) || (minimum(errs) < 1e-10)
      print_tf(@test success)
   end
   println()
end


##

@info("Check Correctness of Multi-species SHIPBasis calculators")

randcoeffs(B) = 2 * (rand(length(B)) .- 0.5) .* (1:length(B)).^(-2)

m_naive_energy(basis::SHIPBasis, at) =
      sum( evaluate(basis, R, at.Z[j], at.Z[i])
            for (i, j, R) in sites(at, cutoff(basis)) )

for basis in ships
   @info("   body-order = $(PoSH.bodyorder(basis))")
   at = bulk(:Si) * 3
   at.Z[:] .= 1
   at.Z[2:2:end] .= 2
   rattle!(at, 0.1)
   print("     energy: ")
   println(@test energy(basis, at) ≈ m_naive_energy(basis, at) )
   print("site-energy: ")
   println(@test energy(basis, at) ≈ sum( site_energy(basis, at, n)
                                         for n = 1:length(at) ) )
   # we can test consistency of forces, site energy etc by taking
   # random inner products with coefficients
   @info("     a few random combinations")
   for n = 1:10
      c = randcoeffs(basis)
      sh = JuLIP.MLIPs.combine(basis, c)
      print_tf(@test energy(sh, at) ≈ dot(c, energy(basis, at)))
      print_tf(@test forces(sh, at) ≈ sum(c*f for (c, f) in zip(c, forces(basis, at))) )
      print_tf(@test site_energy(sh, at, 5) ≈ dot(c, site_energy(basis, at, 5)))
      print_tf(@test site_energy_d(sh, at, 5) ≈ sum(c*f for (c, f) in zip(c, site_energy_d(basis, at, 5))) )
   end
   println()
end


# ----------------------------------------------------------------------
#  Calculator Tests (everything above is a Basis test)
# ----------------------------------------------------------------------

shipsB = ships

@info("--------------- Fast Multi-🚢 Implementation ---------------")

@info("Testing correctness of `SHIP` against `SHIPBasis`")
for B in shipsB
   @info("   bodyorder = $(PoSH.bodyorder(B))")
   coeffs = randcoeffs(B)
   ship = SHIP(B, coeffs)
   @info("   test (de-)dictionisation")
   println(@test decode_dict(Dict(ship)) == ship)
   @show length(B), length(ship)
   tmp = PoSH.alloc_temp(ship, 0)
   @info("      check that SHIPBasis ≈ SHIP")
   for ntest = 1:30
      Rs, Zs, z0 = randR(10, (1,2))
      Es = evaluate!(tmp, ship, Rs, Zs, z0)
      Bs = dot(coeffs, evaluate(B, Rs, Zs, z0))
      print_tf(@test Es ≈ Bs)
   end
   println()
   # ------------------------------------------------------------
   # @info("      Quick timing test")
   # Nr = 30
   # Rs, Zs, z0 = randR(Nr, (1,2))
   # b = PoSH.alloc_B(B)
   # tmp = PoSH.alloc_temp(ship, Nr)
   # tmpB = PoSH.alloc_temp(B, Nr)
   # print("       SHIPBasis : "); @btime evaluate!($b, $tmpB, $B, $Rs, $Zs, $z0)
   # print("            SHIP : "); @btime evaluate!($tmp, $ship, $Rs, $Zs, $z0)
   # println()
end


##
@info("Check Correctness of SHIP gradients")
for B in shipsB
   @info("   body-order = $(PoSH.bodyorder(B))")
   coeffs = randcoeffs(B)
   ship = SHIP(B, coeffs)
   Rs, Zs, z0 = randR(10, (1,2))
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

m_naive_energy(ship::SHIP, at) =
      sum( evaluate(ship, R, at.Z[j], at.Z[i])
            for (i, j, R) in sites(at, cutoff(ship)) )

for B in shipsB
   @info("   body-order = $(PoSH.bodyorder(B))")
   coeffs = randcoeffs(B)
   ship = SHIP(B, coeffs)
   at = bulk(:Si) * 3
   at.Z[1:2:end] .= 1
   at.Z[2:2:end] .= 2
   rattle!(at, 0.1)
   print("     energy: ")
   println(@test energy(ship, at) ≈ m_naive_energy(ship, at) )
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

@info("Multi-species filtering test")
trans = PolyTransform(2, 1.0)
cutf = PolyCutoff2s(2, 0.5, 3.0)
ship = SHIPBasis(SparseSHIP((1, 2), 5,  6; wL = 1.0), trans, cutf, filter=false)
@show maxgrp = maximum(PoSH.alllen_bgrp(ship, 1))
@time fship = PoSH.alg_filter_rpi_basis(ship)
@show length(fship), length(ship)
println(@test length(fship) < length(ship))

end
