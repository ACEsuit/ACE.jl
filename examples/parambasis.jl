
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------



using ACE, JuLIP, Test
using ACE: evaluate
using JuLIP.Testing: print_tf


#---

@info("Construct basic and parameterised basis")

r0 = 2.3
rcut = 5.5
rin = 0.0
maxn = 5
species = [:X ]
D = SparsePSHDegree(wL = 1.0)

trans = ACE.PolyTransform(1, r0)
J = ACE.OrthPolys.transformed_jacobi(10, trans, rcut, rin)
P1 = ACE.RPI.PSH1pBasis(J, maxn, D=D, species = species)

basis = RPIBasis(P1, 3, D, maxn)

J5 = ACE.OrthPolys.transformed_jacobi(5, trans, rcut, rin)
P1basic = ACE.RPI.BasicPSH1pBasis(J5)
basic = RPIBasis(P1basic, 3, D, maxn)

#--- first test: make sure the bases are equivalent

@info("Test bases with and without parameters match")
for ntest = 1:30
   local R, Z, z0 = ACE.Random.rand_nhd(12, J, species)
   print_tf(@test ACE.evaluate(basis, R, Z, z0) ≈ ACE.evaluate(basic, R, Z, z0))
end
println()

#--- second test: perturb parameters

@info("Test basis with perturbed parameters doesn't match (duh...)")
params = basis.pibasis.basis1p.C[1]
basis.pibasis.basis1p.C[1] .+= 0.1 * (rand(size(params)...) .- 0.5)
for ntest = 1:30
   local R, Z, z0 = ACE.Random.rand_nhd(12, J, species)
   print_tf(@test !(ACE.evaluate(basis, R, Z, z0) ≈ ACE.evaluate(basic, R, Z, z0)))
end
println()
