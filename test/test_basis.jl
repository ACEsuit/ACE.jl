
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


@testset "SHIP Basis" begin

##

@info("-------- TEST 🚢  BASIS ---------")
using SHIPs, JuLIP, BenchmarkTools, LinearAlgebra, Test, Random, StaticArrays
using SHIPs: eval_basis!, eval_basis, PolyCutoff1s, PolyCutoff2s
using JuLIP.MLIPs: IPSuperBasis
using JuLIP.Testing: print_tf
using Printf

function randR()
   R = rand(JVecF) .- 0.5
   return (0.9 + 2 * rand()) * R/norm(R)
end
randR(N) = [ randR() for n=1:N ], zeros(Int16, N)
randR(N, syms) = randR(N)[1], rand( Int16.(atomic_number.(syms)), N )
function randiso()
   K = @SMatrix rand(3,3)
   K = K - K'
   Q = rand([-1,1]) * exp(K)
end
function randiso(Rs)
   Q = randiso()
   return [ Q * R for R in shuffle(Rs) ]
end

##

trans = PolyTransform(2, 1.0)
cutf = PolyCutoff2s(2, 0.5, 3.0)

ship2 = SHIPBasis(SparseSHIP(2, 15; wL = 2.0), trans, cutf)
ship3 = SHIPBasis(SparseSHIP(3, 13; wL = 2.0), trans, cutf)
ship4 = SHIPBasis(SparseSHIP(4, 10; wL = 1.5), trans, cutf)
ship5 = SHIPBasis(SparseSHIP(5,  8; wL = 1.5), trans, cutf)
ship6 = SHIPBasis(SparseSHIP(6,  8; wL = 1.5), trans, cutf)
ships = [ship2, ship3, ship4, ship5, ship6]

@info("Test (de-)dictionisation of basis sets")
for ship in ships
   println(@test (decode_dict(Dict(ship)) == ship))
end


@info("Test isometry invariance for 3B-6B 🚢 s")
for ntest = 1:30
   Rs, Zs = randR(20)
   BB = [ eval_basis(🚢, Rs, Zs, 0) for 🚢 in ships ]
   RsX = randiso(Rs)
   BBX = [ eval_basis(🚢, RsX, Zs, 0) for 🚢 in ships ]
   for (B, BX) in zip(BB, BBX)
      print_tf(@test B ≈ BX)
   end
end
println()

##
@info("Test gradients for 3-6B 🚢-basis")
for 🚢 in ships
   @info("  body-order = $(SHIPs.bodyorder(🚢)):")
   Rs, Zs = randR(20)
   tmp = SHIPs.alloc_temp_d(🚢, Rs)
   SHIPs.precompute_grads!(tmp, 🚢, Rs, Zs)
   B1 = eval_basis(🚢, Rs, Zs, 0)
   B = SHIPs.alloc_B(🚢)
   dB = SHIPs.alloc_dB(🚢, Rs)
   SHIPs.eval_basis_d!(B, dB, tmp, 🚢, Rs, Zs, 0)
   @info("      check the basis and basis_d co-incide exactly")
   println(@test B ≈ B1)
   @info("      finite-difference test into random directions")
   for ndirections = 1:20
      Us, Zs = randR(length(Rs))
      errs = Float64[]
      for p = 2:10
         h = 0.1^p
         Bh = eval_basis(🚢, Rs+h*Us, Zs, 0)
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
verbose=false
@info("Test gradients for 3B with and R near the pole")
🚢 = ship2
@info("  body-order = $(SHIPs.bodyorder(🚢)):")
# Rs = [ randR(5); [SVector(1e-14*rand(), 1e-14*rand(), 1.1+1e-6*rand())] ]
Rs = [ randR(5)[1]; [SVector(0, 0, 1.1+0.5*rand())]; [SVector(1e-14*rand(), 1e-14*rand(), 0.9+0.5*rand())] ]
_, Zs = randR(length(Rs))
tmp = SHIPs.alloc_temp_d(🚢, Rs)
SHIPs.precompute_grads!(tmp, 🚢, Rs, Zs)
B1 = eval_basis(🚢, Rs, Zs, 0)
B = SHIPs.alloc_B(🚢)
dB = SHIPs.alloc_dB(🚢, Rs)
SHIPs.eval_basis_d!(B, dB, tmp, 🚢, Rs, Zs, 0)
@info("      finite-difference test into random directions")
for ndirections = 1:30
   Us, _ = randR(length(Rs))
   errs = Float64[]
   for p = 2:10
      h = 0.1^p
      Bh = eval_basis(🚢, Rs+h*Us, Zs, 0)
      dBh = (Bh - B) / h
      dBxU = sum( dot.(Ref(Us[n]), dB[n,:])  for n = 1:length(Rs) )
      push!(errs, norm(dBh - dBxU, Inf))
      verbose && (@printf("  %2d | %.2e \n", p, errs[end]))
   end
   success = (/(extrema(errs)...) < 1e-3) || (minimum(errs) < 1e-10)
   print_tf(@test success)
end
println()


##
@info("Check Correctness of SHIPBasis calculators")

randcoeffs(B) = 2 * (rand(length(B)) .- 0.5) .* (1:length(B)).^(-2)

naive_energy(basis::SHIPBasis, at) =
      sum( eval_basis(basis, R, zeros(Int16, length(R)), 0)
            for (i, j, R) in sites(at, cutoff(basis)) )

for basis in ships
   @info("   body-order = $(SHIPs.bodyorder(basis))")
   at = bulk(:Si) * 3
   at.Z[:] .= 0   # this set of tests is species-agnostic!
   rattle!(at, 0.1)
   print("     energy: ")
   println(@test energy(basis, at) ≈ naive_energy(basis, at) )
   print("site-energy: ")
   println(@test energy(basis, at) ≈ sum( site_energy(basis, at, n)
                                         for n = 1:length(at) ) )
   # we can test consistency of forces, site energy etc by taking
   # random inner products with coefficients
   # TODO [tuple] revive this test after porting `fast`
   # @info("     a few random combinations")
   # for n = 1:10
   #    c = randcoeffs(basis)
   #    sh = JuLIP.MLIPs.combine(basis, c)
   #    print_tf(@test energy(sh, at) ≈ dot(c, energy(basis, at)))
   #    print_tf(@test forces(sh, at) ≈ sum(c*f for (c, f) in zip(c, forces(basis, at))) )
   #    print_tf(@test site_energy(sh, at, 5) ≈ dot(c, site_energy(basis, at, 5)))
   #    print_tf(@test site_energy_d(sh, at, 5) ≈ sum(c*f for (c, f) in zip(c, site_energy_d(basis, at, 5))) )
   # end
   println()
end


end
