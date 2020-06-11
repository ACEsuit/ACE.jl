
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


@testset "Graph-Evaluator" begin

#---

using SHIPs
using Printf, Test, LinearAlgebra, JuLIP, JuLIP.Testing
using JuLIP: evaluate, evaluate_d, evaluate!, evaluate_d!, alloc_temp
using BenchmarkTools

#---

@info("Basic preliminary test for debugging")

basis = SHIPs.Utils.rpi_basis(species = :X, N = 7, maxdeg = 15)
Rs, Zs, z0 = rand_nhd(15, basis.pibasis.basis1p.J, :X)
dEs = zeros(JVecF, length(Rs))

V = SHIPs.Random.randcombine(basis)
tmp = SHIPs.alloc_temp(V, length(Rs));
tmpd = SHIPs.alloc_temp_d(V, length(Rs));

Vtr = SHIPs.GraphPIPot(V)
tmptr = SHIPs.alloc_temp(Vtr, length(Rs))
tmptrd = SHIPs.alloc_temp_d(Vtr, length(Rs));

v = evaluate!(tmp, V, Rs, Zs, z0)
vtr = evaluate!(tmptr, Vtr, Rs, Zs, z0)
println(@test(v ≈ vtr))

dv = evaluate_d(V, Rs, Zs, z0)
dvtr = evaluate_d(Vtr, Rs, Zs, z0)
println(@test(dv ≈ dvtr))


#---

@info("Check several properties of GraphPIPot")
for species in (:X, :Si, [:C, :O], [:C, :O, :H]), N = 1:5
   local Rs, Zs, z0, v, vtr, V, Vtr, basis
   Nat = 15
   basis = SHIPs.Utils.rpi_basis(species = species, N = N, maxdeg = 10)
   Pr = basis.pibasis.basis1p.J
   V = SHIPs.Random.randcombine(basis)
   @info("species = $species; N = $N")
   Vtr = SHIPs.GraphPIPot(V)
   lenpi = maximum(length.(V.pibasis.inner))
   lendag = maximum(length.(Vtr.dags))
   @info("    #$(lenpi) pibasis vs #$(lendag) dag nodes")
   # @info("check (de-)serialisation")
   # println(@test(all(JuLIP.Testing.test_fio(V))))
   @info("Check PiPot and DAGPiPot match")
   for ntest = 1:20
      Rs, Zs, z0 = SHIPs.rand_nhd(Nat, Pr, species)
      v = evaluate(V, Rs, Zs, z0)
      vtr = evaluate(Vtr, Rs, Zs, z0)
      print_tf(@test(v ≈ vtr))
   end
   println()
   # -----------------------
   @info("Check gradients")
   for ntest = 1:20
      Rs, Zs, z0 = SHIPs.rand_nhd(Nat, Pr, species)
      V0 = evaluate(Vtr, Rs, Zs, z0)
      dV0 = evaluate_d(Vtr, Rs, Zs, z0)
      Us = [ rand(eltype(Rs)) .- 0.5 for _=1:length(Rs) ]
      dV0_dUs = sum(transpose.(dV0) .* Us)
      errs = []
      for p = 2:12
         h = 0.1^p
         V_h = evaluate(Vtr, Rs + h * Us, Zs, z0)
         dV_h = (V_h - V0) / h
         # @show norm(dAA_h - dAA_dUs, Inf)
         push!(errs, norm(dV_h - dV0_dUs, Inf))
      end
      success = (/(extrema(errs)...) < 1e-3) || (minimum(errs) < 1e-10)
      print_tf(@test success)
   end
   println()

end

#---

end
