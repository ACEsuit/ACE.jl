
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
function randrot()
   K = @SMatrix rand(3,3)
   K = K - K'
   Q = rand([-1,1]) * exp(K)
end
function randrot(Rs)
   Q = randrot()
   return [ Q * R for R in shuffle(Rs) ]
end

##

trans3 = PolyTransform(3, 1.0)
ship3 = SHIPBasis(3, 15, 2.0, trans3, 2, 0.5, 3.0)
trans2 = PolyTransform(2, 1.3)
ship2 = SHIPBasis(2, 15, 2.0, trans2, 2, 0.5, 3.0)
ships = [ship2, ship3]

Rs = randR(20)
BB = [ eval_basis(🚢, Rs) for 🚢 in ships ]

@info("Test rotational invariance for 3B and 4B 🚢 s")
for ntest = 1:10
   RsX = randrot(Rs)
   BBX = [ eval_basis(🚢, RsX) for 🚢 in ships ]
   for (B, BX) in zip(BB, BBX)
      print((@test B ≈ BX), " ")
   end
end
println()


# ## 
# Rs = randR(10)
# RsX = randrot(Rs)
# B = eval_basis(ship3, Rs)
# BX = eval_basis(ship3, RsX)
# I = findall( abs.(B-BX) .> 1e-10 )
# ship3.Nu[I]
# BX[I] - B[I]
# length(I)
#

end
