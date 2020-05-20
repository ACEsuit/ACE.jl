
##

using SHIPs
using Printf, Test, LinearAlgebra, JuLIP, JuLIP.Testing
using JuLIP: evaluate, evaluate_d, evaluate!, evaluate_d!, alloc_temp
using BenchmarkTools, StaticArrays
using Profile

##

@info("Basic preliminary test for debugging")

basis = SHIPs.Utils.rpi_basis(species = :X, N = 6, maxdeg = 10)
V = SHIPs.Random.randcombine(basis)
Rs, Zs, z0 = rand_nhd(15, basis.pibasis.basis1p.J, :X)
tmp = SHIPs.alloc_temp(V, length(Rs));
Vtr = SHIPs.Tree.TreePIPot(V)
tmptr = SHIPs.alloc_temp(Vtr, length(Rs))

v = evaluate!(tmp, V, Rs, Zs, z0)
vtr = evaluate!(tmptr, Vtr, Rs, Zs, z0)
println(@test(v ≈ vtr))

##

@info("Check several properties of PIPotential")
for species in (:Si, [:C, :O], [:C, :O, :H]), N = 1:5
   Nat = 15
   basis = SHIPs.Utils.rpi_basis(species = species, N = N, maxdeg = 10)
   Pr = basis.pibasis.basis1p.J
   V = SHIPs.Random.randcombine(basis)
   @info("species = $species; N = $N")
   Vtr = SHIPs.Tree.TreePIPot(V)
   lenpi = maximum(length.(V.pibasis.inner))
   lentree = maximum(length.(Vtr.trees))
   @info("    #$(lenpi) pibasis vs #$(lentree) tree nodes")
   # @info("check (de-)serialisation")
   # println(@test(all(JuLIP.Testing.test_fio(V))))
   @info("Check basis and potential match")
   for ntest = 1:20
      Rs, Zs, z0 = SHIPs.rand_nhd(Nat, Pr, species)
      v = evaluate(V, Rs, Zs, z0)
      vtr = evaluate(Vtr, Rs, Zs, z0)
      print_tf(@test(v ≈ vtr))
   end
   println()
end

##
