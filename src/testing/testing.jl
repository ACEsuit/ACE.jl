

module Testing

using Test
import ACE

ACE.@extimports
ACE.@aceimports

import ACEbase
import ACEbase.Testing: print_tf, test_fio

export print_tf, test_fio, test_transform



# include("testmodel.jl")
# include("testlsq.jl")



# # ---------- code for consistency tests
#
# test_basis(D::Dict) = ACE.Utils.rpi_basis(;
#                species = Symbol.(D["species"]), N = D["N"],
#                maxdeg = D["maxdeg"],
#                r0 = D["r0"], rcut = D["rcut"],
#                D = ACE.RPI.SparsePSHDegree(wL = D["wL"]) )
#
#
# _evaltest(::Val{:E}, V, at) = energy(V, at)
# _evaltest(::Val{:F}, V, at) = vec(forces(V, at))
#
# function createtests(V, ntests; tests = ["E", "F"], kwargs...)
#    testset = Dict[]
#    for n = 1:ntests
#       at = ACE.Random.rand_config(V; kwargs...)
#       D = Dict("at" => write_dict(at), "tests" => Dict())
#       for t in tests
#          D["tests"][t] = _evaltest(Val(Symbol(t)), V, at)
#       end
#       push!(testset, D)
#    end
#    return testset
# end
#
#
# function runtests(V, tests; verbose = true)
#    for test in tests
#       at = read_dict(test["at"])
#       for (t, val) in test["tests"]
#          print_tf( @test( val ≈ _evaltest(Val(Symbol(t)), V, at) ) )
#       end
#    end
# end




# ---------- code for transform tests

import ForwardDiff
import ACE: evaluate, evaluate_d, inv_transform

function test_transform(T, rrange, ntests = 100)

   rmin, rmax = extrema(rrange)
   rr = rmin .+ rand(100) * (rmax-rmin)
   xx = evaluate.(Ref(T), rr)
   # check syntactic sugar
   xx1 = T.(rr)
   print_tf(@test xx1 == xx)
   # check inversion
   rr1 = inv_transform.(Ref(T), xx)
   print_tf(@test rr1 ≈ rr)
   # check gradient
   dx = evaluate_d.(Ref(T), rr)
   adx = ForwardDiff.derivative.(Ref(r -> evaluate(T, r)), rr)
   print_tf(@test dx ≈ adx)

   # TODO: check that the transform doesn't allocate
   @allocated begin
      x = 0.0;
      for r in rr
         x += evaluate(T, r)
      end
   end
end


# --------- Code for derivative tests


import Base.*

struct __TestSVec{T}
   val::T
end

*(a::Number, u::__TestSVec) = a * u.val
*(a::SMatrix, u::__TestSVec) = a * u.val
*(a::SArray{Tuple{N1,N2,N3}}, u::__TestSVec) where {N1, N2,N3} =
      reshape(reshape(a, Size(N1*N2, N3)) * u.val, Size(N1, N2))


end
