
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
