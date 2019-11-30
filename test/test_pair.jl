
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


@testset "Pair Potentials" begin

@info("-------- TEST PAIR  BASIS ---------")
using PoSH, JuLIP, BenchmarkTools, LinearAlgebra, Test, Random, StaticArrays
using PoSH: PairBasis

at = bulk(:W, cubic=true) * 3
rattle!(at, 0.03)
r0 = rnn(:W)
X = copy(positions(at))

trans = PolyTransform(2, 1.3)
pB = PairBasis(10, trans, 2, 2.1*r0)

@info("test (de-)dictionisation of PairBasis")
println(@test decode_dict(Dict(pB)) == pB)

E = energy(pB, at)
DE = - forces(pB, at)

@info("Finite-difference test on PairBasis forces")
for ntest = 1:20
   U = rand(JVecF, length(at)) .- 0.5
   DExU = dot.(DE, Ref(U))
   errs = Float64[]
   for p = 2:10
      h = 0.1^p
      Eh = energy(pB, set_positions!(at, X+h*U))
      DEhxU = (Eh-E) / h
      push!(errs, norm(DExU - DEhxU, Inf))
   end
   success = (/(extrema(errs)...) < 1e-3) || (minimum(errs) < 1e-10)
   print_tf(@test success)
end
println()

end
