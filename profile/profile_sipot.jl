
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------


using ACE, JuLIP, BenchmarkTools

#---

fname = "/Users/ortner/Dropbox/Public/SHIPPots/Si_split_1.2_reg_test_v05.json"
D = load_dict(fname)
V = ACE.Import.import_pipot_v05(D)

at = bulk(:Si, cubic=true) * 3;
@show length(at)

#---

println("length(PIBasis) = ")
@show length(V.pibasis.inner[1])

#---

@info("Standard Evaluator")


@btime energy($V, $at)
@btime forces($V, $at)

print("time/atom (energy): ");
println(" $((@timed energy(V, at)).time / length(at) * 1000) ms")

print("time/atom (forces): ");
println(" $((@timed forces(V, at)).time / length(at) * 1000) ms")


#---

@info("Graph Evaluator")
println("(it takes a long time to generate the graph - sorry!")
Vgr = ACE.GraphPIPot(V)
println("Number of nodes")
@show length(Vgr.dags[1].nodes)

@btime energy($Vgr, $at)
@btime forces($Vgr, $at)
print("time/atom (energy): ");
println(" $((@timed energy(Vgr, at)).time / length(at) * 1000) ms")
print("time/atom (forces): ");
println(" $((@timed forces(Vgr, at)).time / length(at) * 1000) ms")
