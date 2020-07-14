
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------

@assert length(ARGS) in [0, 1]

using SHIPs, JuLIP


testfile = @__DIR__() * "/regtest.json"

# prepare the basis sets and coefficients
# (potentials will be generated from scratch)
# if there is a parameter file use it, otherwise generate a fresh one
if !(isfile(testfile))
   @info("No test file exists. Generating...")
   D = SHIPs.Testing.generate_regtests()
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
      basis = SHIPs.Testing.test_basis(params)
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
   run(pipeline(`git rev-parse HEAD`, stdout = buf))
   label = String(take!(buf))[1:end-1]
end

# run and save the actual tests
SHIPs.Testing.run_regression_tests!(D, label)
JuLIP.save_dict(testfile, D)
