

using PoSH, Printf, Test, LinearAlgebra
using PoSH: PolyTransform, rbasis, eval_basis, eval_basis_d
using PoSH.JacobiPolys: Jacobi
using PyPlot

trans = PolyTransform(1, 1)
B2 = rbasis(4, trans, 2, 0.5, 2.0)

rr = range(0, 2.5, length=100)
BB = zeros(100, length(B2))
for (i, r) in enumerate(rr)
   BB[i, :] = eval_basis(B2, r)
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
