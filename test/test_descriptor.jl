
@testset "PoSH Descriptor" begin


@info("-------- TEST 🚢 DESCRIPTOR ---------")
@info("I'm really just checking that the constructor and evaluator run ok")
using PoSH, JuLIP, Test, ASE
using PoSH.Descriptors

try
   desc = SHIPDescriptor(:Si, deg = 5, rcut = 5.0)
   at = bulk("Si", cubic=true) * 2
   B1 = descriptors(desc, at)
   println(@test true)
catch
   println(@test false)
end

end
