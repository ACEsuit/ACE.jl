
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------


@testset "PolyPairPot" begin

@info("--------------- PolyPairPot Implementation ---------------")

##

using ACE
using Printf, Test, LinearAlgebra, JuLIP, JuLIP.Testing
using JuLIP: evaluate, evaluate_d
using JuLIP.Potentials: i2z, numz
using JuLIP.MLIPs: combine

randr() = 1.0 + rand()
randcoeffs(B) = rand(length(B)) .* (1:length(B)).^(-2)

##

maxdeg = 8
r0 = 1.0
rcut = 3.0

trans = PolyTransform(1, r0)
Pr = transformed_jacobi(maxdeg, trans, rcut; pcut = 2)
pB = ACE.PairPotentials.PolyPairBasis(Pr, :W)
coeffs = randcoeffs(pB)
V = combine(pB, coeffs)



##

at = bulk(:W, cubic=true) * 3
rattle!(at, 0.03)
r0 = rnn(:W)
X = copy(positions(at))
energy(V, at)

@info("Testing correctness of `PolyPairPot` against `PolyPairBasis`")
@info("    test `combine`")
coeffs = randcoeffs(pB)
V = combine(pB, coeffs)
println(@test energy(V, at) ≈ sum(V.coeffs .*  energy(pB, at)))

##
@info("   test (de-)dictionisation")
println(@test all(JuLIP.Testing.test_fio(V)))

@info("      check that PolyPairBasis ≈ PolyPairPot")
for ntest = 1:10
   rattle!(at, 0.01)
   coeffs = randcoeffs(pB)
   V = combine(pB, coeffs)

   E_V = energy(V, at)
   E_b = dot(energy(pB, at), coeffs)
   print_tf(@test E_V ≈ E_b)

   F_V = forces(V, at)
   F_b = sum(coeffs .* forces(pB, at))
   print_tf(@test F_V ≈ F_b)

   V_V = virial(V, at)
   V_b = sum(coeffs .* virial(pB, at))
   print_tf(@test V_V ≈ V_b)
end
println()

##

@info("      Standard JuLIP Force Consistency Test")
variablecell!(at)
rattle!(at, 0.03)
println(@test JuLIP.Testing.fdtest(V, at))

##
end
