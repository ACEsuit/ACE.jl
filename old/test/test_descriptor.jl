
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------


@testset "ACE Descriptor" begin


@info("-------- TEST 🚢 DESCRIPTOR ---------")
@info("I'm really just checking that the constructor and evaluator run ok")
using ACE, JuLIP, Test, ASE
using ACE.Descriptors

try
   desc = SHIPDescriptor(:Si, deg = 5, rcut = 5.0)
   at = bulk("Si", cubic=true) * 2
   B1 = descriptors(desc, at)
   println(@test true)
catch
   println(@test false)
end

end
