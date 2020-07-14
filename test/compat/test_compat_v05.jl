
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------



@testset "Compatibility" begin

#---

using SHIPs, Printf, Test, LinearAlgebra, JuLIP, JuLIP.Testing
using JuLIP: evaluate, evaluate_d
using SHIPs: PIPotential

#---

function compat_tests(V::PIPotential, rtests, tests)
   Jr = V.pibasis.basis1p.J

   @info("Running radial basis tests")
   if !isempty(rtests)
      for n = 1:length(rtests)
         r = rtests[1]["r"]
         Jr_test = rtests[1]["Pr"]
         Jr_new = evaluate(Jr, r)
         print_tf(@test Jr_test ≈ Jr_new)
      end
      println()
   end

   @info("Running potential tests")
   if !isempty(tests)
      for t1 in tests
         Rs = JVecF.(t1["Rs"])
         Zs = AtomicNumber.(t1["Zs"])
         z0 = AtomicNumber(t1["z0"])
         valold = t1["val"]
         valnew = evaluate(V, Rs, Zs, z0)
         print_tf(@test valold ≈ valnew)
      end
      println()
   end
end


##

# a randomly generated single-species potential
fname = @__DIR__() * "/models/v05/randship_v05.json"
D = load_dict(fname)
V = SHIPs.Import.import_pipot_v05(D)
compat_tests(V, D["rtests"], D["tests"])

##

# Cas' Si fit from v0.5.x times
# See if the Si potential lives in the dropbox, otherwise skip this test
sifile = "/users/ortner/Dropbox/Public/SHIPPots/Si_split_1.2_reg_test_v05.json"
if isfile(sifile)
   D = load_dict(sifile)
   V = SHIPs.Import.import_pipot_v05(D)
   compat_tests(V, D["rtests"], D["tests"])
end

##


end
