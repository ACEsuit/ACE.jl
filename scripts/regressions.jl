
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------

@assert length(ARGS) in [0, 1]

using ACE, JuLIP, InteractiveUtils, ProgressMeter
using JuLIP.MLIPs: combine
using BenchmarkTools: @belapsed

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


#---


testfile = @__DIR__() * "/regtest.json"

if !(isfile(testfile))
   error("No test file exists. Generating...")
end

# load the parameter file - keep it open for the rest of this
D = JuLIP.FIO.load_dict(testfile)

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
run_regression_tests!(D, label)
JuLIP.save_dict(testfile, D)
