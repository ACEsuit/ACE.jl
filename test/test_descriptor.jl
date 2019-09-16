
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


@testset "SHIPs Descriptor" begin


@info("-------- TEST 🚢 DESCRIPTOR ---------")
using SHIPs, JuLIP, Test, ASE
using SHIPs.Descriptors

try
   desc = SHIPDescriptor(:Si, deg = 5, rcut = 5.0)
   at = bulk("Si", cubic=true) * 2
   B1 = descriptors(desc, at)
   @test true
catch
   @test false
end

end
