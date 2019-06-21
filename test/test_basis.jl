
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


@testset "SHIP Basis" begin

@info("-------- TEST 🚢  BASIS ---------")
using SHIPs, JuLIP, BenchmarkTools, LinearAlgebra, Test, Random, StaticArrays
using SHIPs: eval_basis!, eval_basis

function randR()
   R = rand(JVecF) .- 0.5
   return (0.9 + 2 * rand()) * R/norm(R)
end
randR(N) = [ randR() for n=1:N ]
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

trans3 = PolyTransform(3, 1.0)
ship3 = SHIPBasis(3, 13, 2.0, trans3, 2, 0.5, 3.0)
trans2 = PolyTransform(2, 1.3)
ship2 = SHIPBasis(2, 15, 2.0, trans2, 2, 0.5, 3.0)
ship4 = SHIPBasis(4, 11, 1.0, trans3, 2, 0.5, 3.0)
ships = [ship2, ship3, ship4]

@info("Test isometry invariance for 3B, 4B and 5B 🚢 s")
for ntest = 1:20
   Rs = randR(20)
   BB = [ eval_basis(🚢, Rs) for 🚢 in ships ]
   RsX = randiso(Rs)
   BBX = [ eval_basis(🚢, RsX) for 🚢 in ships ]
   for (B, BX) in zip(BB, BBX)
      print_tf(@test B ≈ BX)
   end
end
println()


@info("Test gradients for 3B, 4B and 5B 🚢 s")

Rs = randR(20)
🚢 = ships[1]
store = SHIPs.alloc_temp_d(🚢, Rs)
SHIPs.precompute_grads!(store, 🚢, Rs)
B1 = eval_basis(🚢, Rs)
B, dB = SHIPs.alloc_dB(🚢, Rs)
SHIPs.eval_basis_d!(B, dB, 🚢, Rs, store)

end
