
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


using SHIPs, JuLIP, BenchmarkTools


trans = PolyTransform(2, 1.0)
ships = [SHIPBasis(SparseSHIPBasis(15, 2.0), n, trans, 2, 0.5, 3.0) for n = 2:4]

Rs = 1.0 .+ rand(JVecF, 50)

@info("profile precomputation of A")
@btime SHIPs.precompute_A!($(ships[1]), $Rs)
@btime SHIPs.precompute_A!($(ships[1]), $Rs)

@info("profile basis computation")
for n = 2:4
   @info("  body-order $(n+1):")
   🚢 = ships[n-1]
   B = SHIPs.alloc_B(🚢)
   @info("     eval_basis:")
   @btime SHIPs.eval_basis!($B, $🚢, $Rs)
   @btime SHIPs.eval_basis!($B, $🚢, $Rs)
   @info("     eval_basis_d:")
   dB = SHIPs.alloc_dB(🚢, Rs)
   store = SHIPs.alloc_temp_d(🚢, Rs)
   @btime SHIPs.eval_basis_d!($B, $dB, $🚢, $Rs, $store)
   @btime SHIPs.eval_basis_d!($B, $dB, $🚢, $Rs, $store)
end

# ##
# using Profile
# 🚢 = ships[1]
# B = SHIPs.alloc_B(🚢)
# dB = SHIPs.alloc_dB(🚢, Rs)
# store = SHIPs.alloc_temp_d(🚢, Rs)
# @btime SHIPs.eval_basis_d!($B, $dB, $🚢, $Rs, $store)
#
# ##
#
# function runn(N, args...)
#    for n = 1:N
#       SHIPs.eval_basis_d!(args...)
#    end
# end
# runn(2, B, dB, 🚢, Rs, store)
#
# Profile.clear()
# @profile runn(10_000,  B, dB, 🚢, Rs, store)
#
# Profile.print()
