
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------



@testset "Compatibility" begin

##

using SHIPs, Printf, Test, LinearAlgebra, JuLIP, JuLIP.Testing
using JuLIP: evaluate, evaluate_d
using SHIPs: PIPotential

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
   end
end

##

fname = @__DIR__() * "/models/randship_v05.json"
D = load_dict(fname)
V = SHIPs.Import.import_pipot_v05(D)
compat_tests(V, D["rtests"], D["tests"])

##


end
