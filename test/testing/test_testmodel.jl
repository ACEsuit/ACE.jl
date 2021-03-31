
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
