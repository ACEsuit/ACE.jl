
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------


using JSON, PrettyTables, Printf, Statistics

#---

testfile = @__DIR__() * "/regtest.json"
D = JSON.parsefile(testfile)

tol = 0.05
new = "head"
old = "v0.7.0"
keys = ["evalpot", "gradpot", "evalbasis", "gradbasis"]

#---
summary = Dict( [k => [] for k in keys ]... )
failed = []

for (params, t) in zip(D["paramsets"], D["results"])
   curt = Dict()
   for s in keys
      r = t[new][s] / t[old][s]
      push!(summary[s], r)
      curt[s] = r
   end
   if any(collect(values(curt)) .> 1 + tol)
      curt["params"] = params
      push!(failed, curt)
   end
end

#---



println("---------------------------------")
println("      Regression Test Summary")
println("---------------------------------")
println("   Failed Tests: $(length(failed))")

table = vcat([ hcat([ f(summary[k]) for k in keys ]...)
               for f in [minimum, maximum, mean, std] ]...)
pretty_table( [ ["min", "max", "mean", "std"] round.(table, digits=3)],
               vcat(["STATS"], keys) )

println("Report on Failed Tests:")
println("-"^80)
for t in failed
   println("Parameters:")
   print("  ")
   for (k, v) in t["params"]
      print(k, ": ", v, "; ")
   end
   println()
   println("Regression:")
   print("  ")
   for k in keys
      print(k, ": ", round( t[k], digits=3 ), "; " )
   end
   println()
   println("-"^80)
end
