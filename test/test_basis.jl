
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


using SHIPs, JuLIP, BenchmarkTools

trans = PolyTransform(2, 1.0)
ship = SHIPBasis(3, 15, 1.5, trans, 2, 0.5, 3.0)

Rs = 1.0 .+ rand(JVecF, 30)
@btime SHIPs.precompute_A!($ship, $Rs)

length(ship.A)


ship = SHIPBasis(5, 15, 2.0, trans, 2, 0.5, 3.0)
SHIPs.length_B(ship)
