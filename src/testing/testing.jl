
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------



module Testing

using Test
import ACE
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
include("../aceimports.jl")

include("testmodel.jl")
include("testlsq.jl")


# ---------- code for consistency tests

test_basis(D::Dict) = ACE.Utils.rpi_basis(;
               species = Symbol.(D["species"]), N = D["N"],
               maxdeg = D["maxdeg"],
               r0 = D["r0"], rcut = D["rcut"],
               D = ACE.RPI.SparsePSHDegree(wL = D["wL"]) )


_evaltest(::Val{:E}, V, at) = energy(V, at)
_evaltest(::Val{:F}, V, at) = vec(forces(V, at))

function createtests(V, ntests; tests = ["E", "F"], kwargs...)
   testset = Dict[]
   for n = 1:ntests
      at = ACE.Random.rand_config(V; kwargs...)
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




# ---------- code for transform tests

import ForwardDiff
import ACE.Transforms: transform, transform_d, inv_transform

function test_transform(T, rrange, ntests = 100)

   rmin, rmax = extrema(rrange)
   rr = rmin .+ rand(100) * (rmax-rmin)
   xx = [ transform(T, r) for r in rr ]
   # check syntactic sugar
   xx1 = [ T(r) for r in rr ]
   print_tf(@test xx1 == xx)
   # check inversion
   rr1 =  inv_transform.(Ref(T), xx)
   print_tf(@test rr1 ≈ rr)
   # check gradient
   dx = transform_d.(Ref(T), rr)
   adx = ForwardDiff.derivative.(Ref(r -> transform(T, r)), rr)
   print_tf(@test dx ≈ adx)

   # TODO: check that the transform doesn't allocate
   @allocated begin
      x = 0.0;
      for r in rr
         x += transform(T, r)
      end
   end
end



end
