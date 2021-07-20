@testset "GNLModel"  begin

##
using LinearAlgebra: length
using ACE, ACEbase
using Printf, Test, LinearAlgebra, ACE.Testing, Random
using ACE: evaluate, evaluate_d, SymmetricBasis, NaiveTotalDegree, PIBasis
using ACEbase.Testing: fdtest
using BenchmarkTools
using StaticArrays
using Zygote

##

@info("General NonLin model tests")
    
# construct the 1p-basis
D = NaiveTotalDegree()
maxdeg = 6
ord = 3

B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, D = D)

# generate a configuration
nX = 54
Xs = rand(EuclideanVectorState, B1p.bases[1], nX)
cfg = ACEConfig(Xs)

φ = ACE.Invariant()
pibasis = PIBasis(B1p, ord, maxdeg; property = φ)
basis = SymmetricBasis(pibasis, φ)
       
BB = evaluate(basis, cfg)

#Non Linearity
#fun(ϕ) = ϕ[1] + sqrt((1/10)^2 + abs(ϕ[2])) - 1/10 # Finnis Sinclair
fun(ϕ) = ϕ[1] + sqrt((1/10)^2 + abs(ϕ[2])) + ϕ[3]*ϕ[4] #other non-linearity
Nϕ = 4 #number of ϕ, important that it corresponds to the non-linearity
c = [rand(length(BB)).-0.5 for i = 1:Nϕ]

GNL = ACE.Models.NLModel(basis, c, evaluator = :standard, F = fun)

# # ------------------------------------------------------------------------
# #    Finite difference test
# # ------------------------------------------------------------------------

@info("FD test grad_param")
outer_nonL(x) = 2*x + cos(x)
tEval = ACE.Models.EVAL_NL(GNL,cfg)
test_fe(θ) = outer_nonL(tEval(θ))
cfg = ACEConfig(Xs)
for ntest = 1:30
    c_tst = [rand(length(BB)).-0.5 for i = 1:Nϕ]
    F = t ->  test_fe(c_tst + [reshape(t,(length(BB),Nϕ))[:,i] for i = 1:Nϕ])
    dF = t -> collect(Iterators.flatten(Zygote.gradient(test_fe,c_tst)[1]))
    print_tf(@test fdtest(F, dF, zeros(length(collect(Iterators.flatten(c_tst)))), verbose=false))
end
println()

@info("FD test grad_config")
for ntest = 1:30
    Us = randn(SVector{3, Float64}, length(Xs))
    F = t ->  ACE.Models.EVAL_NL(GNL,ACEConfig(Xs + t[1] * Us))(c)
    dF = t -> [sum([ dot(u, g) for (u, g) in zip(Us, ACE.Models.EVAL_D_NL(GNL,ACEConfig(Xs + t[1] * Us))(c))])]
    print_tf(@test fdtest(F, dF, [0.0], verbose=false))
end
println()

using JuLIP

emt = JuLIP.EMT()
at = bulk(:Cu, cubic=true) * 3
rattle!(at,0.6)

@info("Energies test")
outer_nonL(x) = 2*x + cos(x)
en = ACE.Models.ENERGY_NL(GNL, at, emt)
e_me(θ) = outer_nonL(en(θ))

for ntest = 1:3
    c_tst = [rand(length(BB)).-0.5 for i = 1:Nϕ]
    F = t ->  e_me(c_tst + [reshape(t,(length(BB),Nϕ))[:,i] for i = 1:Nϕ])
    dF = t -> [collect(Iterators.flatten(Zygote.gradient(e_me,c_tst)[1]))[i] for i in 1:length(c_tst[:])]
    print_tf(@test fdtest(F, dF, zeros(length(collect(Iterators.flatten(c_tst)))), verbose=false))
end
println()

@info("Forces test")
fr = ACE.Models.FORCES_NL(GNL, at, emt)
f_me(θ) = sum(sum(fr(θ)))

for ntest = 1:3
    c_tst = [rand(length(BB)).-0.5 for i = 1:Nϕ]
    F = t ->  f_me(c_tst + [reshape(t,(length(BB),Nϕ))[:,i] for i = 1:Nϕ])
    dF = t -> [collect(Iterators.flatten(Zygote.gradient(f_me,c_tst)[1]))[i] for i in 1:length(c_tst[:])]
    print_tf(@test fdtest(F, dF, zeros(length(collect(Iterators.flatten(c_tst)))), verbose=false))
end
println()

# # ------------------------------------------------------------------------
# #    Pseudo model test
# # ------------------------------------------------------------------------

@info("Loss function test")

emt = JuLIP.EMT()
at1 = bulk(:Cu, cubic=true) * 3
rattle!(at1,0.6)
at2 = bulk(:Cu, cubic=true) * 3
rattle!(at2,0.6)
at3 = bulk(:Cu, cubic=true) * 3
rattle!(at3,0.6)

Efun = [ACE.Models.ENERGY_NL(GNL, at1, emt),ACE.Models.ENERGY_NL(GNL, at2, emt),ACE.Models.ENERGY_NL(GNL, at3, emt)]
Ffun = [ACE.Models.FORCES_NL(GNL, at1, emt),ACE.Models.FORCES_NL(GNL, at2, emt),ACE.Models.FORCES_NL(GNL, at3, emt)]

Et = [energy(emt,at1),energy(emt,at2),energy(emt,at3)]
Ft = [forces(emt,at1),forces(emt,at2),forces(emt,at3)]

we = 1.0
wf = 1.0
L_FS(θ) = sum([we^2*sum(abs2,Efun[i](θ) - Et[i]) + wf^2*sum(abs2,sum(Ffun[i](θ) - Ft[i])) for i in 1:3])/3

for ntest = 1:3
    c_tst = [rand(length(BB)).-0.5 for i = 1:Nϕ]
    F = t ->  L_FS(c_tst + [reshape(t,(length(BB),Nϕ))[:,i] for i = 1:Nϕ])
    dF = t -> [collect(Iterators.flatten(Zygote.gradient(L_FS,c_tst)[1]))[i] for i in 1:length(c_tst[:])]
    print_tf(@test fdtest(F, dF, zeros(length(collect(Iterators.flatten(c_tst)))), verbose=false))
end
println()

end
