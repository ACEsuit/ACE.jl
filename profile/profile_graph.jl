
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------


using ACE, JuLIP, BenchmarkTools

#---



basis = ACE.Utils.rpi_basis(; species=:Si, N = 5, maxdeg = 14)
@show length(basis)
V = ACE.Random.randcombine(basis)

#---

at = bulk(:Si, cubic=true) * 5
nlist = neighbourlist(at, cutoff(V))
tmp = JuLIP.alloc_temp(V, maxneigs(nlist))
tmp_d = JuLIP.alloc_temp_d(V, maxneigs(nlist))
F = zeros(JVecF, length(at))
@code_warntype JuLIP.energy!(tmp, V, at)
@code_warntype JuLIP.forces!(F, tmp_d, V, at)

@btime JuLIP.forces!($F, $tmp_d, $V, $at)
@btime JuLIP.forces($V, $at)
