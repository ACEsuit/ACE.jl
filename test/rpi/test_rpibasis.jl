
@testset "RPIBasis"  begin

##


using SHIPs
using Random, Printf, Test, LinearAlgebra, JuLIP, JuLIP.Testing
using JuLIP: evaluate, evaluate_d


##

@info("Basic test of RPIBasis construction and evaluation")
maxdeg = 4
N =  2
r0 = 1.0
rcut = 3.0
trans = PolyTransform(1, r0)
Pr = transformed_jacobi(maxdeg, trans, rcut; pcut = 2)
D = SHIPs.SparsePSHDegree()
P1 = SHIPs.BasicPSH1pBasis(Pr; species = :X, D = D)

##

# pibasis = SHIPs.PIBasis(P1, 2, D, maxdeg)
rpibasis = SHIPs.RPIBasis(P1, N, D, maxdeg)


##
rotc = SHIPs.Rotations3D.Rot3DCoeffs()
A2B = SHIPs._rpi_A2B_matrix(rotc, rpibasis.pibasis, 1
   ) |> Matrix
LinearAlgebra.rank(A2B)

# spec = collect(keys(rpibasis.pibasis.inner[1].b2iAA))
# display(spec)

##
using StaticArrays
cc = SHIPs.Rotations3D.Rot3DCoeffs()

##

ll = SVector(1,1,1,1)
nn = SVector(2,1,0,3)
zz = SVector(0,0,0,0)

println("-----------")
Uri = SHIPs.Rotations3D.ri_basis(cc, ll)
display(Uri)
println("-----------")

Urpi, Ms = SHIPs.Rotations3D.rpi_basis(cc, zz, nn, ll)
display(Urpi)
println("-----------")

##
# check single-species
Nat = 15
Rs, Zs, z0 = SHIPs.rand_nhd(Nat, Pr, :X)
AA = evaluate(basis, Rs, Zs, z0)
println(@test(length(basis, z0) == length(AA)))

# check multi-species
maxdeg = 5
Pr = transformed_jacobi(maxdeg, trans, rcut; pcut = 2)
species = [:C, :O, :H]
P1 = SHIPs.BasicPSH1pBasis(Pr; species = [:C, :O, :H], D = D)
basis = SHIPs.PIBasis(P1, 3, D, maxdeg)
Rs, Zs, z0 = SHIPs.rand_nhd(Nat, Pr, species)
AA = evaluate(basis, Rs, Zs, z0)
println(@test(length(basis, z0) == length(AA)))

##

@info("Check permutation invariance")
for species in (:X, :Si, [:C, :O, :H])
   Nat = 15
   P1 = SHIPs.BasicPSH1pBasis(Pr; species = species)
   basis = SHIPs.PIBasis(P1, 3, D, maxdeg)
   for ntest = 1:10
      Rs, Zs, z0 = SHIPs.rand_nhd(Nat, Pr, species)
      p = randperm(length(Rs))
      print_tf(@test(evaluate(basis, Rs, Zs, z0) ≈
                     evaluate(basis, Rs[p], Zs[p], z0)))
   end
end
println()

##

end
