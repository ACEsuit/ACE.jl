
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


@testset "OrthogonalPolynomials" begin

@info("--------- Testing OrthogonalPolynomials ----------")

##
using SHIPs, Test

using LinearAlgebra: norm, cond
using ForwardDiff

using SHIPs.JacobiPolys: Jacobi
using SHIPs.OrthPolys: OrthPolyBasis

using JuLIP: evaluate, evaluate_d
using JuLIP.Testing: print_tf

##

@info("Discretised Jacobi are close to the real Jacobi Poly's")

N = 15
Nquad = 1000
dt =  2 / Nquad
tdf = range(-1.0+dt/2, 1.0-dt/2, length=Nquad)
Jd = OrthPolyBasis(N,  0, 1.0, 0, -1.0, tdf)
J = Jacobi(0.0, 0.0, N-1, normalise=true)

for ntest = 1:30
   x = 2*rand() - 1
   Jdx = evaluate(Jd, x)
   Jx = evaluate(J, x)
   Jx /= Jx[1]
   print_tf((@test norm(Jx - Jdx, Inf) < 10/N^2))
end
println()

##
@info("de-dictionisation")

for ntest = 1:10
   N = 8
   Nquad = 1000
   tdf = rand(1000)
   ww = 1.0 .+ rand(1000)
   Jd = OrthPolyBasis(N, 2, 1.0, 1, -1.0, tdf, ww)

   print_tf(@test decode_dict(Dict(Jd)) == Jd)

   tmpf = tempname() * ".json"
   save_json(tmpf, Dict(Jd))
   print_tf(@test decode_dict(load_json(tmpf)) == Jd)
end

##
@info("Construction and FD vs Grad for randomly generated OrthPolyBasis")

N = 8
Nquad = 1000
tdf = rand(1000)
ww = 1.0 .+ rand(1000)
Jd = OrthPolyBasis(N, 2, 1.0, 2, -1.0, tdf, ww)

let errtol = 1e-12, ntest = 100
   nfail = 0
   for itest = 1:ntest
      x = 2*rand() - 1
      dJx = evaluate_d(Jd, x)
      adJx = ForwardDiff.derivative(x -> evaluate(Jd, x), x)
      err = maximum(abs.(dJx - adJx) ./ (1.0 .+ abs.(dJx)))
      if err > errtol
         nfail += 1
      end
   end
   println((@test nfail == 0))
end
##

@info("Testing TransformedPolys")

trans = PolyTransform(2, 1.0)
Pnew = SHIPs.OrthPolys.transformed_jacobi(10, trans, 2.0, 0.5; pcut = 2, pin = 2)
fcut = PolyCutoff2s(2, 0.5, 2.0)
Pold = SHIPs.TransformedJacobi(9, trans, fcut)

# r = 2*rand() + 0.25
# pnew = evaluate(Pnew, r)
# pold = evaluate(Pold, r)
# norm((pold[1]/pnew[1]) * abs.(pnew) - pold, Inf)

@info("   ... consistency with old implementation")
for ntest = 1:30
   r = 2*rand() + 0.25
   pnew = evaluate(Pnew, r)
   pold = evaluate(Pold, r)
   if r <= 0.5 || r >= 2.0
      print_tf(@test( norm(pnew, Inf) == norm(pold, Inf) == 0 ))
   else
      pnew_scal = pold[1]/pnew[1] * pnew
      print_tf(@test norm(abs.(pnew_scal) - abs.(pold), Inf) < 1e-10 )
   end
end
println()

##

@info("   ... consistency of derivatives")
for ntest = 1:30
   r = 2*rand() + 0.25
   dp = evaluate_d(Pnew, r)
   if r <= 0.5 || r >= 2.0
      print_tf(@test( norm(dp, Inf) == 0 ))
   else
      adp = ForwardDiff.derivative(x -> evaluate(Pnew, x), r)
      print_tf(@test norm(dp - adp) < 1e-12)
   end
end
println()

##

@info("Testing the orthogonality (via A-basis)")

spec = SparseSHIP(3, 10)
P = SHIPs.OrthPolys.transformed_jacobi(SHIPs.maxK(spec)+1, trans, 2.0, 0.5;
                                      pcut = 2, pin = 2)
shpB = SHIPBasis(spec, P)

function evalA(shpB, tmp, Rs)
   Zs = zeros(Int16, length(Rs))
   SHIPs.precompute_A!(tmp, shpB, Rs, Zs, 1)
   return tmp.A[1]
end

function A_gramian(shpB, Nsamples = 100_000)
   tmp = SHIPs.alloc_temp(shpB)
   lenA = length(tmp.A[1])
   G = zeros(ComplexF64, lenA, lenA)
   for n = 1:Nsamples
      R = SHIPs.Utils.rand(shpB.J)
      A = evalA(shpB, tmp, [R])
      for i = 1:lenA, j = 1:lenA
         G[i,j] +=  A[i] * A[j]'
      end
   end
   return G
end

G = A_gramian(shpB)
println(@test cond(G) < 1.2)

##

end

# ## Quick look at the basis
# using Plots
# N = 5
# Jd = SHIPs.OrthPolys.discrete_jacobi(N; pcut = 3, pin = 2)
# tp = range(-1, 1, length=100)
# Jp = zeros(length(tp), N)
# for (i,t) in enumerate(tp)
#    Jp[i, :] = evaluate(Jd, t)
# end
# plot(tp, Jp)
