



@info("--------- Testing OrthogonalPolynomials ----------")

##
using ACE, Test, ForwardDiff

using LinearAlgebra: norm, cond
using ACE.OrthPolys: OrthPolyBasis
using ACE: evaluate, evaluate_d, evaluate_ed 
using ACEbase.Testing: print_tf


##
@info("de-dictionisation")

for ntest = 1:10
   local N, Nquad, tdf, ww, Jd 
   N = 8
   Nquad = 1000
   tdf = rand(1000)
   ww = 1.0 .+ rand(1000)
   Jd = OrthPolyBasis(N, 2, 1.0, 1, -1.0, tdf, ww)
   print_tf(@test all(ACE.Testing.test_fio(Jd)))
end
println()


##
@info("Construction and FD vs Grad for randomly generated OrthPolyBasis")

N = 8
Nquad = 1000
tdf = rand(1000)
ww = 1.0 .+ rand(1000)
Jd = OrthPolyBasis(N, 2, 1.0, 2, -1.0, tdf, ww)

let errtol = 1e-12, ntest = 50
   for itest = 1:ntest
      x = 2*rand() - 1
      Jx = evaluate(Jd, x)
      dJx = evaluate_d(Jd, x)
      Jx1, dJx1 = evaluate_ed(Jd, x)
      print_tf(@test all( (Jx, dJx) .≈ (Jx1, dJx1) ) )
      adJx = ForwardDiff.derivative(x -> evaluate(Jd, x), x)
      err = maximum(abs.(dJx - adJx) ./ (1.0 .+ abs.(dJx)))
      print_tf(@test (err < errtol))
   end
   println()
end




##

# TODO: replace this test with an orthogonality test
# @info("Discretised Jacobi are close to the real Jacobi Poly's")
#
# N = 15
# Nquad = 1000
# dt =  2 / Nquad
# tdf = range(-1.0+dt/2, 1.0-dt/2, length=Nquad)
# Jd = OrthPolyBasis(N,  0, 1.0, 0, -1.0, tdf)
# J = Jacobi(0.0, 0.0, N-1, normalise=true)
#
# for ntest = 1:30
#    x = 2*rand() - 1
#    Jdx = evaluate(Jd, x)
#    Jx = evaluate(J, x)
#    Jx /= Jx[1]
#    print_tf((@test norm(Jx - Jdx, Inf) < 10/N^2))
# end
# println()

##

# TODO: convert into a quadrature orthogonality test
# @info("Testing the orthogonality (via A-basis)")
#
# spec = SparseSHIP(3, 10)
# P = ACE.OrthPolys.transformed_jacobi(ACE.maxK(spec)+1, trans, 2.0, 0.5;
#                                       pcut = 2, pin = 2)
# aceB = SHIPBasis(spec, P)
#
# function evalA(aceB, tmp, Rs)
#    Zs = zeros(Int16, length(Rs))
#    ACE.precompute_A!(tmp, aceB, Rs, Zs, 1)
#    return tmp.A[1]
# end
#
# function A_gramian(aceB, Nsamples = 100_000)
#    tmp = ACE.alloc_temp(aceB)
#    lenA = length(tmp.A[1])
#    G = zeros(ComplexF64, lenA, lenA)
#    for n = 1:Nsamples
#       R = ACE.rand_vec(aceB.J)
#       A = evalA(aceB, tmp, [R])
#       for i = 1:lenA, j = 1:lenA
#          G[i,j] +=  A[i] * A[j]'
#       end
#    end
#    return G
# end
#
# G = A_gramian(aceB)
# println(@test cond(G) < 1.2)

##


# ## Quick look at the basis
# using Plots
# N = 5
# Jd = ACE.OrthPolys.discrete_jacobi(N; pcut = 3, pin = 2)
# tp = range(-1, 1, length=100)
# Jp = zeros(length(tp), N)
# for (i,t) in enumerate(tp)
#    Jp[i, :] = evaluate(Jd, t)
# end
# plot(tp, Jp)
