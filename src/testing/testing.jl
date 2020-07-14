
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


module Testing

using Test
import SHIPs
import InteractiveUtils


using BenchmarkTools: @belapsed
using LinearAlgebra: eigvals, eigen
using ProgressMeter: @showprogress

import JuLIP.Potentials: F64fun
import JuLIP: Atoms, bulk, rattle!, positions, energy, forces, JVec,
              chemical_symbol, mat, rnn,
              read_dict, write_dict
import JuLIP.MLIPs: combine
import JuLIP.Testing: print_tf

include("../extimports.jl")
include("../shipimports.jl")

include("testmodel.jl")
include("testlsq.jl")


# ---------- code for consistency tests

test_basis(D::Dict) = SHIPs.Utils.rpi_basis(;
               species = Symbol.(D["species"]), N = D["N"],
               maxdeg = D["maxdeg"],
               r0 = D["r0"], rcut = D["rcut"],
               D = SHIPs.RPI.SparsePSHDegree(wL = D["wL"]) )


_evaltest(::Val{:E}, V, at) = energy(V, at)
_evaltest(::Val{:F}, V, at) = vec(forces(V, at))

function createtests(V, ntests; tests = ["E", "F"], kwargs...)
   testset = Dict[]
   for n = 1:ntests
      at = SHIPs.Random.rand_config(V; kwargs...)
      D = Dict("at" => write_dict(at), "tests" => Dict())
      for t in tests
         D["tests"][t] = _evaltest(Val(Symbol(t)), V, at)
      end
      push!(testset, D)
   end
   return testset
end


function runtests(V, tests; verbose = true)
   for test in tests
      at = read_dict(test["at"])
      for (t, val) in test["tests"]
         print_tf( @test( val ≈ _evaltest(Val(Symbol(t)), V, at) ) )
      end
   end
end



# ---------- code for regression tests



function generate_regtests()
   degrees = Dict(2 => [6, 7, 8],
                  3 => [7, 9, 11],
                  4 => [8, 10, 12])
   # degrees = Dict(2 => [8, 13, 18],
   #                4 => [8, 13, 18],
   #                6 => [10, 16, 18]),
   #                8 => [10, 13, 16])
   species = ["Si"]
   r0 = rnn(:Si)
   rcuts = [5.5, 7.0]
   params = []
   for (N, degs) in degrees, maxdeg in degs, rcut in rcuts
      push!( params, Dict( "species" => species, "N" => N,
                           "maxdeg" => maxdeg, "r0" => r0,
                           "rcut" => rcut, "wL" => 1.3 ) )
   end
   return Dict("paramsets" => params,
               "at" => write_dict(SHIPs.Random.rand_config(:Si; repeat=3)))
end


function run_regression_tests!(D::Dict, label::AbstractString)
   # store meta-data: save the label
   if label in D["labels"]
      error("label $(label) already exists in the regression tests")
   end
   push!(D["labels"], label)
   # julia version and machine
   buf = IOBuffer()
   InteractiveUtils.versioninfo(buf)
   D["versioninfo"][label] = String(take!(buf))
   # read the benchmark configuration
   at = read_dict(D["at"])
   # run the actual benchmarks
   @showprogress for test in D["results"]
      basis = read_dict(test["basis"])
      tBE, tBF = run_regtest(basis, at)
      V = combine(basis, Float64.(test["coeffs"]))
      tVE, tVF = run_regtest(V, at)
      test[label] = Dict("evalbasis" => tBE, "gradbasis" => tBF,
                         "evalpot" => tVE, "gradpot" => tVF)
   end
end


function run_regtest(calc, at)
   return (@belapsed energy($calc, $at)),
          (@belapsed forces($calc, $at))
end


end
