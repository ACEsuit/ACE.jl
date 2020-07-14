
using Test
import SHIPs, JuLIP
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

# include("../src/extimports.jl")
include("../src/shipimports.jl")

#---

testfile = @__DIR__() * "/regtest.json"

if isfile(testfile)
   error("regressions file `regtest.json` already exists. Aborting...")
end

# generate the set of parameters
# degrees = Dict(2 => [6, 7, 8],
#                3 => [7, 9, 11],
#                4 => [8, 10, 12])
degrees = Dict(2 => [8, 13, 18],
               4 => [8, 13, 18],
               6 => [10, 16, 18]),
               8 => [10, 13, 16])
species = ["Si"]
r0 = rnn(:Si)
rcuts = [5.5, ]
params = []
for (N, degs) in degrees, maxdeg in degs, rcut in rcuts
   push!( params, Dict( "species" => species, "N" => N,
                        "maxdeg" => maxdeg, "r0" => r0,
                        "rcut" => rcut, "wL" => 1.3 ) )
end
D = Dict("paramsets" => params,
               "at" => write_dict(SHIPs.Random.rand_config(:Si; repeat=3)))


# generate and save the basis sets if they don't  exist yet
@info("Generate basis and coefficients...")
D["results"] = []
D["labels"] = []
D["versioninfo"] = Dict{String, Any}()
@showprogress for params in D["paramsets"]
   basis = SHIPs.Testing.test_basis(params)
   coeffs = SHIPs.Random.randcoeffs(basis; diff=2)
   push!(D["results"], Dict("basis" => write_dict(basis), "coeffs" => coeffs))
end
JuLIP.save_dict(testfile, D)
