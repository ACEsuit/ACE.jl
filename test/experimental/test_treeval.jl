
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


##

using SHIPs
using Printf, Test, LinearAlgebra, JuLIP, JuLIP.Testing
using JuLIP: evaluate, evaluate_d, evaluate!, evaluate_d!, alloc_temp
using BenchmarkTools, StaticArrays
using Profile

##

@info("Basic preliminary test for debugging")

basis = SHIPs.Utils.rpi_basis(species = :X, N = 6, maxdeg = 10)
Rs, Zs, z0 = rand_nhd(15, basis.pibasis.basis1p.J, :X)
dEs = zeros(JVecF, length(Rs))

V = SHIPs.Random.randcombine(basis)
tmp = SHIPs.alloc_temp(V, length(Rs));
tmpd = SHIPs.alloc_temp_d(V, length(Rs));

Vtr = SHIPs.Tree.TreePIPot(V)
tmptr = SHIPs.alloc_temp(Vtr, length(Rs))
tmptrd = SHIPs.alloc_temp_d(Vtr, length(Rs));

##

@btime evaluate_d!($dEs, $tmpd, $V, $Rs, $Zs, $z0)
@btime evaluate_d!($dEs, $tmptrd, $Vtr, $Rs, $Zs, $z0)

##
v = evaluate!(tmp, V, Rs, Zs, z0)
vtr = evaluate!(tmptr, Vtr, Rs, Zs, z0)
println(@test(v ≈ vtr))

##

dv = evaluate_d(V, Rs, Zs, z0)
dvtr = evaluate_d(Vtr, Rs, Zs, z0)

dv ≈ dvtr

##
#
# @info("Check several properties of PIPotential")
# for species in (:Si, [:C, :O], [:C, :O, :H]), N = 1:5
#    Nat = 15
#    basis = SHIPs.Utils.rpi_basis(species = species, N = N, maxdeg = 10)
#    Pr = basis.pibasis.basis1p.J
#    V = SHIPs.Random.randcombine(basis)
#    @info("species = $species; N = $N")
#    Vtr = SHIPs.Tree.TreePIPot(V)
#    lenpi = maximum(length.(V.pibasis.inner))
#    lentree = maximum(length.(Vtr.trees))
#    @info("    #$(lenpi) pibasis vs #$(lentree) tree nodes")
#    # @info("check (de-)serialisation")
#    # println(@test(all(JuLIP.Testing.test_fio(V))))
#    @info("Check basis and potential match")
#    for ntest = 1:20
#       Rs, Zs, z0 = SHIPs.rand_nhd(Nat, Pr, species)
#       v = evaluate(V, Rs, Zs, z0)
#       vtr = evaluate(Vtr, Rs, Zs, z0)
#       print_tf(@test(v ≈ vtr))
#    end
#    println()
# end
#
# ##
