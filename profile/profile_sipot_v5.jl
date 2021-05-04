
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------


using ACE, JuLIP, BenchmarkTools

#---

fname = "/Users/ortner/Dropbox/Public/SHIPPots/Si_split_1.2_reg_test_v05.json"
D = load_dict(fname)
V = decode_dict(D)

#---

at = bulk(:Si, cubic=true) * 3;
@show length(at)

println("Energy:")
@btime energy($V, $at)
println("Forces:")
@btime forces($V, $at)

#---
print("time/atom (energy): ");
println(" $((@timed energy(V, at)).time / length(at) * 1000) ms")

print("time/atom (forces): ");
println(" $((@timed forces(V, at)).time / length(at) * 1000) ms")
