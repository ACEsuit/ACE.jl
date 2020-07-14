
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------

@assert length(ARGS) in [0, 1]

using SHIPs, JuLIP

module SHIPs_Regressions

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

include("../src/extimports.jl")
include("../src/shipimports.jl")

test_basis(D::Dict) = SHIPs.Utils.rpi_basis(;
               species = Symbol.(D["species"]), N = D["N"],
               maxdeg = D["maxdeg"],
               r0 = D["r0"], rcut = D["rcut"],
               D = SHIPs.RPI.SparsePSHDegree(wL = D["wL"]) )

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


#---


testfile = @__DIR__() * "/regtest.json"

# prepare the basis sets and coefficients
# (potentials will be generated from scratch)
# if there is a parameter file use it, otherwise generate a fresh one
if !(isfile(testfile))
   @info("No test file exists. Generating...")
   D = SHIPs_Regressions.generate_regtests()
   JuLIP.save_dict(testfile, D)
end

# load the parameter file - keep it open for the rest of this
D = JuLIP.FIO.load_dict(testfile)

# generate and save the basis sets if they don't  exist yet
if !haskey(D, "results")
   @info("Generate basis and coefficients...")
   D["results"] = []
   D["labels"] = []
   D["versioninfo"] = []
   for params in D["paramsets"]
      basis = SHIPs_Regressions.test_basis(params)
      coeffs = SHIPs.Random.randcoeffs(basis; diff=2)
      push!(D["results"], Dict("basis" => write_dict(basis), "coeffs" => coeffs))
   end
   JuLIP.save_dict(testfile, D)
end

# create a label for the tests
if length(ARGS) == 1
   label = ARGS[1]
else
   # get the current git hash and use it as label
   buf = IOBuffer()
   run(pipeline(`git rev-parse --short HEAD`, stdout = buf))
   label = String(take!(buf))[1:end-1]
end

# run and save the actual tests
SHIPs_Regressions.run_regression_tests!(D, label)
JuLIP.save_dict(testfile, D)
