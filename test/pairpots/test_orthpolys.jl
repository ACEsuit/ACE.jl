
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


@testset "OrthogonalPolynomials" begin

@info("--------- Testing OrthogonalPolynomials ----------")

##
using PoSH, Test

using LinearAlgebra: norm

using PoSH.JacobiPolys: Jacobi
using PoSH.OrthPolys: OrthPolyBasis

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

let h = 1e-4, errtol = 1e-4, ntest = 100, allowedfail = 5
   nfail = 0
   for itest = 1:ntest
      x = 2*rand() - 1
      dJx = evaluate_d(Jd, x)
      dhJx = (evaluate(Jd, x + h) - evaluate(Jd, x-h)) / (2*h)
      err = maximum(abs.(dJx - dhJx) ./ (1.0 .+ abs.(dJx)))
      if err > errtol
         nfail += 1
      end
   end
   @info("nfail = $nfail (out of $ntest)")
   println((@test nfail <= allowedfail))
end
##

end

# ## Quick look at the basis
# using Plots
# N = 5
# Jd = PoSH.OrthPolys.discrete_jacobi(N; pcut = 3, pin = 2)
# tp = range(-1, 1, length=100)
# Jp = zeros(length(tp), N)
# for (i,t) in enumerate(tp)
#    Jp[i, :] = evaluate(Jd, t)
# end
# plot(tp, Jp)
