


#---

using ACE
using Printf, Test, LinearAlgebra, JuLIP, JuLIP.Testing, Random
using JuLIP: evaluate, evaluate_d, evaluate_ed
using ACE: combine
using BenchmarkTools

#---

maxdeg = 13
N = 4
species = [:Ti, :Al]
r0 = 1.0
rcut = 3.0
trans = PolyTransform(1, r0)
Pr = transformed_jacobi(maxdeg, trans, rcut; pcut = 2)
D = ACE.SparsePSHDegree()
P1 = ACE.BasicPSH1pBasis(Pr; species = species)
basis = ACE.PIBasis(P1, N, D, maxdeg)

#---

c = randcoeffs(basis)
p = 0.05
csp = copy(c)
for i = 1:length(csp)
   if rand() > p
      csp[i] = 0
   end
end

#---

Vdag = combine(basis, csp)
V = standardevaluator(Vdag)
Vdag_sp = ACE.deletezeros(Vdag)
V_sp = standardevaluator(Vdag_sp)
# Vdag_sp = graphevaluator(V_sp)

#---

Nat = 3
Rs, Zs, z0 = ACE.rand_nhd(Nat, Pr, species)

# evaluate_d(Vdag_sp, Rs, Zs, z0)

val0 = evaluate(V, Rs, Zs, z0)
dV0 = evaluate_d(V, Rs, Zs, z0)
for (V, str) in ( (Vdag, "Vdag"), (V_sp, "V_sp"), (Vdag_sp, "Vdag_sp") )
   @info("Error for $str:")
   val1 = evaluate(V, Rs, Zs, z0)
   @show abs(val0 - val1)
   dV1 = evaluate_d(V, Rs, Zs, z0)
   @show norm(dV0 - dV1)
end

#---

for (pot, name) in ( (V, "V"), (Vdag, "Vdag"),
                   (V_sp, "V_sp"), (Vdag_sp, "Vdag_sp") )
   tmp = ACE.alloc_temp(pot, Nat)
   @info("Energy Timing for $name")
   @btime ACE.evaluate!($tmp, $pot, $Rs, $Zs, $z0)
end

#---

graphevaluator(Vdag_sp)

for (pot, name) in ( (V, "V"), (Vdag, "Vdag"),
                   (V_sp, "V_sp"), (Vdag_sp, "Vdag_sp") )
   tmpd = ACE.alloc_temp_d(pot, Nat)
   @info("Force Timing for $name")
   @btime ACE.evaluate_d!($(tmpd.dV), $tmpd, $pot, $Rs, $Zs, $z0)
end
