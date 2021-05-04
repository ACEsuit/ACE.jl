
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------



using ACE, Printf, Test, LinearAlgebra
using ACE: PolyTransform, rbasis,
using JuLIP: evaluate, evaluate_d
using ACE.JacobiPolys: Jacobi
using PyPlot

trans = PolyTransform(1, 1)
B2 = rbasis(4, trans, 2, 0.5, 2.0)

rr = range(0, 2.5, length=100)
BB = zeros(100, length(B2))
for (i, r) in enumerate(rr)
   BB[i, :] = evaluate(B2, r)
end

##
clf()
for i = 1:length(B2)
   plot(rr, BB[:,i], label = "P_$i")
end
legend()
xlabel("r")
ylabel("P_i(r) = J_i(x) * w(x)")

PyPlot.display_figs()
