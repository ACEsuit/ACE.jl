

using SHIPs, JuLIP, Test
using SHIPs: evaluate
using JuLIP.Testing: print_tf


#---

@info("Construct basic and parameterised basis")

r0 = 2.3
rcut = 5.5
rin = 0.0
maxn = 5
species = [:X ]
D = SparsePSHDegree(wL = 1.0)

trans = SHIPs.PolyTransform(1, r0)
J = SHIPs.OrthPolys.transformed_jacobi(10, trans, rcut, rin)
P1 = SHIPs.RPI.PSH1pBasis(J, maxn, D=D, species = species)

basis = RPIBasis(P1, 3, D, maxn)

J5 = SHIPs.OrthPolys.transformed_jacobi(5, trans, rcut, rin)
P1basic = SHIPs.RPI.BasicPSH1pBasis(J5)
basic = RPIBasis(P1basic, 3, D, maxn)

#--- first test: make sure the bases are equivalent

@info("Test bases with and without parameters match")
for ntest = 1:30
   local R, Z, z0 = SHIPs.Random.rand_nhd(12, J, species)
   print_tf(@test SHIPs.evaluate(basis, R, Z, z0) ≈ SHIPs.evaluate(basic, R, Z, z0))
end
println()

#--- second test: perturb parameters

@info("Test basis with perturbed parameters doesn't match (duh...)")
params = basis.pibasis.basis1p.C[1]
basis.pibasis.basis1p.C[1] .+= 0.1 * (rand(size(params)...) .- 0.5)
for ntest = 1:30
   local R, Z, z0 = SHIPs.Random.rand_nhd(12, J, species)
   print_tf(@test !(SHIPs.evaluate(basis, R, Z, z0) ≈ SHIPs.evaluate(basic, R, Z, z0)))
end
println()
