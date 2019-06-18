
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


using SHIPs, JuLIP, BenchmarkTools, LinearAlgebra, Test, Random, StaticArrays
using SHIPs: eval_basis!, eval_basis

trans3 = PolyTransform(3, 1.0)
ship3 = SHIPBasis(3, 15, 2.0, trans3, 2, 0.5, 3.0)
trans2 = PolyTransform(2, 1.3)
ship2 = SHIPBasis(2, 15, 2.0, trans2, 2, 0.5, 3.0)

ships = [ship2, ship3]

function randR()
   R = rand(JVecF) .- 0.5
   return (0.9 + 2 * rand()) * R/norm(R)
end
randR(N) = [ randR() for n=1:N ]

Rs = randR(20)
BB = [ eval_basis(🚢, Rs) for 🚢 in ships ]

for ntest = 1:10
   K = @SMatrix rand(3,3)
   K = K - K'
   Q = exp(K)
   RsX = [ Q * R for R in shuffle(Rs) ]
   BBX = [ eval_basis(🚢, RsX) for 🚢 in ships ]
   for (B, BX) in zip(BB, BBX)
      @show norm(B - BX, Inf)
      # println(@test B ≈ BX)
   end
end
