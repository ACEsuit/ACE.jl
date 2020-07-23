
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


@testset "SHIPs Descriptor" begin

##

@info("-------- TEST 🚢 DESCRIPTOR ---------")
@info("I'm really just checking that the constructor and evaluator run ok")
using SHIPs, JuLIP, Test, ASE
using SHIPs.Descriptors

##

@info("standard descriptor (sparse)")
try
   desc = SHIPDescriptor(:Si, deg = 10, rcut = 5.0)
   @show length(desc)
   at = bulk("Si", cubic=true) * 2
   B1 = descriptors(desc, at)
   println(@test true)
catch
   println(@test false)
end

##

@info("manual descriptor (tensor)")
try
   #                           bo  lmax  nmax
   spec = SHIPs.TensorSHIP(:Si, 3,   4,   4)
   desc = SHIPDescriptor(:Si, spec=spec, rcut = 5.0)
   @show length(desc)
   at = bulk("Si", cubic=true) * 2
   B1 = descriptors(desc, at)
   println(@test true)
catch
   println(@test false)
end

##

end
