
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------


@testset "TestModel" begin

#---
##

using Test, ACE, JuLIP
using ACE.Testing: ToyModel

#---

@info("Finite difference test of toymodel with different setups")
for species in ( [:Fe, ], [:Fe, :Al], [:Fe, :Ti, :Al] )
   @info("    species = $species")
   V = ToyModel(species)
   at = rand(V, nrepeat=3)
   energy(V, at)
   F = forces(V, at)
   println(@test JuLIP.Testing.fdtest(V, at))
end

#---
##



end
