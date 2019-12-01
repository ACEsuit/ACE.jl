
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------

@testset "PolyPairBasis" begin

@info("-------- Test PolyPairBasis Implementation ---------")

##

using Test
using PoSH, JuLIP, LinearAlgebra, Test
using JuLIP.Testing, JuLIP.MLIPs
randr() = 1.0 + rand()
randcoeffs(B) = rand(length(B)) .* (1:length(B)).^(-2)

##

at = bulk(:W, cubic=true) * 3
rattle!(at, 0.03)
r0 = rnn(:W)
X = copy(positions(at))

trans = PolyTransform(2, r0)
fcut = PolyCutoff1s(2, 0.5*r0, 2.5*r0)
pB = PolyPairBasis(:W, 10, trans, fcut)

##

@info("test (de-)dictionisation of PairBasis")
println(@test decode_dict(Dict(pB)) == pB)

E = energy(pB, at)
DE = - forces(pB, at)

##

@info("Finite-difference test on PolyPairBasis forces")
# for ntest = 1:20
U = [ (rand(JVecF) .- 0.5) for _=1:length(at) ]
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
# end
println()
##

end


@testset "PolyPairPot" begin

@info("--------------- PolyPairPot Implementation ---------------")

##

using PoSH, JuLIP, LinearAlgebra, Test
using JuLIP.Testing, LinearAlgebra
using JuLIP.MLIPs
randr() = 1.0 + rand()
randcoeffs(B) = rand(length(B)) .* (1:length(B)).^(-2)

##

at = bulk(:W, cubic=true) * 3
rattle!(at, 0.03)
r0 = rnn(:W)

trans = PolyTransform(2, r0)
fcut = PolyCutoff2s(2, 0.5*r0, 1.95*r0)
B = PolyPairBasis(:W, 10, trans, fcut)

##


@info("Testing correctness of `PolyPairPot` against `PolyPairBasis`")

@info("    test `combine`")
coeffs = randcoeffs(B)
pot = combine(B, coeffs)
println(@test pot == PolyPairPot(B, coeffs))


##
@info("   test (de-)dictionisation")
println(@test decode_dict(Dict(pot)) == pot)

@info("      check that PolyPairBasis ≈ PolyPairPot")
for ntest = 1:30
   rattle!(at, 0.01)

   E_pot = energy(pot, at)
   E_b = dot(energy(B, at), coeffs)
   print_tf(@test E_pot ≈ E_b)

   F_pot = forces(pot, at)
   F_b = sum(coeffs .* forces(B, at))
   print_tf(@test F_pot ≈ F_b)

   V_pot = virial(pot, at)
   V_b = sum(coeffs .* virial(B, at))
   print_tf(@test V_pot ≈ V_b)
end
println()

@info("      Standard JuLIP Consistency Test")
variablecell!(at)
rattle!(at, 0.03)
JuLIP.Testing.fdtest(pot, at)

##
end
