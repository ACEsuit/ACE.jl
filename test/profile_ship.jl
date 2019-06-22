
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


using SHIPs, JuLIP, BenchmarkTools, LinearAlgebra

function randR()
   R = rand(JVecF) .- 0.5
   return (0.9 + 2 * rand()) * R/norm(R)
end
randR(N) = [ randR() for n=1:N ]
randcoeffs(B) = rand(length(B)) .* (1:length(B)).^(-2)

trans = PolyTransform(2, 1.0)
BB = [SHIPBasis(n, 15, 2.0, trans, 2, 0.5, 3.0) for n = 2:4]

Rs = randR(50)
@info("profile precomputation of A")
@btime SHIPs.precompute_A!($(BB[1]), $Rs)
@btime SHIPs.precompute_A!($(BB[1]), $Rs)

@info("profile basis and ship computation")
for n = 2:4
   @info("  body-order $(n+1):")
   Rs = randR(50)
   B = BB[n-1]
   coeffs = randcoeffs(B)
   🚢 = SHIP(B, coeffs)
   b = SHIPs.alloc_B(B)
   store = SHIPs.alloc_temp(🚢)
   @info("     evaluate a site energy:")
   print("         SHIPBasis: "); @btime SHIPs.eval_basis!($b, $B, $Rs, nothing)
   print("         SHIP     : "); @btime SHIPs.evaluate!($🚢, $Rs, $store)

   store = SHIPs.alloc_temp_d(🚢, Rs)
   dEs = zeros(JVecF, length(Rs))
   db = SHIPs.alloc_dB(B, Rs)
   dbtmp = SHIPs.alloc_temp_d(B, Rs)

   @info("     site energy gradient:")
   store = SHIPs.alloc_temp_d(🚢, Rs)
   print("         SHIPBasis: "); @btime SHIPs.eval_basis_d!($b, $db, $B, $Rs, $dbtmp)
   print("         SHIP     : "); @btime SHIPs.evaluate_d!($dEs, $🚢, $Rs, $store)
end
