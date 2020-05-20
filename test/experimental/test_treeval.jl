
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


basis = SHIPs.Utils.rpi_basis(N = 3, maxdeg = 6)
length(basis)
length(basis.pibasis)
V = SHIPs.Random.randcombine(basis)
Rs, Zs, z0 = rand_nhd(15, basis.pibasis.basis1p.J)
tmp = SHIPs.alloc_temp(V, length(Rs))
evaluate!(tmp, V, Rs, Zs, z0)
# @btime evaluate!($tmp, $V, $Rs, $Zs, $z0);

##

Vtr = SHIPs.Tree.TreePIPot(V)

tmptr = alloc_temp(Vtr, length(Rs))
evaluate!(tmptr, Vtr, Rs, Zs, z0)
