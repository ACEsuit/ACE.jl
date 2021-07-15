
#@testset "GfsModel"  begin

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

@info("Basic test of GFS model construction and evaluation")
    
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
c = rand(length(BB),2) .- 0.5

fun = ACE.Models.FinnisSinclair(1/10)
#fun = ACE.Models.ToyExp()

standard = ACE.Models.GfsModel(basis, c, evaluator = :standard, F = fun) 

# ## FIO

# @info("Check FIO")
# using ACEbase.Testing: test_fio 
# println(@test(all(test_fio(standard; warntype = false))))

# ##

#evaluate_ref(basis, cfg, c, F) = F([sum(evaluate(basis, cfg) .* c[:,1]).val, sum(evaluate(basis, cfg) .* c[:,2]).val])

#test evaluation, ref vs zygote vs evaluate
#print_tf(@test(evaluate_ref(basis,cfg,c,fun) ≈ ACE.Models.EVAL_me(standard,cfg)(c) ≈ evaluate(standard, cfg)))

# ------------------------------------------------------------------------
#    Finite difference test
# ------------------------------------------------------------------------

@info("FD test grad_config")
for ntest = 1:30
    Us = randn(SVector{3, Float64}, length(Xs))
    F = t ->  evaluate(standard, ACEConfig(Xs + t[1] * Us))
    dF = t -> [sum([ dot(u, g) for (u, g) in zip(Us, ACE.Models.grad_config(standard,ACEConfig(Xs + t[1] * Us))) ])]
    print_tf(@test fdtest(F, dF, [0.0], verbose=false))
end
println()


@info("FD test grad_param")
cfg = ACEConfig(Xs)
for ntest = 1:30
    c_tst = rand(length(BB),2) .- 0.5
    F = t ->  evaluate(ACE.Models.set_params!(standard,c_tst[:] + t),cfg)
    dF = t -> [collect(Iterators.flatten(ACE.Models.grad_params(ACE.Models.set_params!(standard,c_tst[:] + t),cfg)))[i].val for i in 1:length(c_tst[:])]
    print_tf(@test fdtest(F, dF, zeros(length(c_tst)), verbose=false))
end
println()

outer_nonL(x) = 2*x + cos(x)

@info("FD test grad_param with zygote")
cfg = ACEConfig(Xs)
for ntest = 1:30
    c_tst = rand(length(BB),2) .- 0.5
    F = t ->  outer_nonL(evaluate(ACE.Models.set_params!(standard,c_tst[:] + t), cfg))
    evalme = ACE.Models.EVAL_me(standard,cfg)
    test_fe(θ) = outer_nonL(evalme(θ))
    dF = t -> [collect(Iterators.flatten(Zygote.gradient(test_fe,c_tst[:] + t)[1]))[i].val for i in 1:length(c_tst[:])]
    print_tf(@test fdtest(F, dF, zeros(length(c_tst)), verbose=false))
end
println()

#@info("FD test grad_params_config")

"""
first-order finite-difference test for scalar F
```julia
fdtest(F, dF, x; h0 = 1.0, verbose=true)
```
"""
function fdtest_θX(F, dF, x::AbstractVector, y::AbstractVector; h0 = 1.0, verbose=true)
   errors = Float64[]
   dE = dF(x,y)
   xn = deepcopy(x)
   yn = deepcopy(y)
   # loop through finite-difference step-lengths
   verbose && @printf("---------|----------- \n")
   verbose && @printf("    h    | error \n")
   verbose && @printf("---------|----------- \n")
   for p = 2:11
      h = 0.1^p
      dEh = copy(dE)
      for n = 1:length(dE)
         x[n] += h
         y[1] += h
         xn[n] -= h
         yn[1] -= h

         fpp = F(x,y)
         fnn = F(xn,yn)
         fpn = F(x,yn)
         fnp = F(xn,y)
         @show (fpp - fpn - fnp + fnn)
         @show 4*h^2
         dEh[n] = (fpp - fpn - fnp + fnn) / 4*h^2
 
         x[n] -= h
         y[1] -= h
         xn[n] += h
         yn[1] += h
      end
      print(kal)
      @show norm(dEh, Inf)
      @show norm(dE, Inf)
      push!(errors, norm(dE - dEh, Inf))
      verbose && @printf(" %1.1e | %4.2e  \n", h, errors[end])
   end
   verbose && @printf("---------|----------- \n")
   if minimum(errors) <= 1e-3 * maximum(errors)
      verbose && println("passed")
      return true
   else
      @warn("""It seems the finite-difference test has failed, which indicates
      that there is an inconsistency between the function and gradient
      evaluation. Please double-check this manually / visually. (It is
      also possible that the function being tested is poorly scaled.)""")
      return false
   end
end


# for ntest = 1:30
#     c_tst = rand(length(BB),2) .- 0.5
#     Us = randn(SVector{3, Float64}, length(Xs))
#     F = (t,s) ->  evaluate(ACE.Models.set_params!(standard,c_tst[:] + t), ACEConfig(Xs + s[1] * Us))  
#     function dF(t,s)
#         a = ACE.Models.grad_params_config(ACE.Models.set_params!(standard,c_tst[:] + t), ACEConfig(Xs + s[1] * Us))
#         b = collect(Iterators.flatten([[sum([ dot(u, g) for (u, g) in zip(Us, a[1][i,:]) ]) for i in  1:length(a[1][:,1])],[sum([ dot(u, g) for (u, g) in zip(Us, a[2][i,:]) ]) for i in  1:length(a[2][:,1])]]))
#         return(b)
#     end
#     print_tf(@test fdtest_θX(F, dF, zeros(length(c_tst)), [0.0], verbose=true))
# end
# println()


# print_tf(@test fdtest(F, dF, zeros(length(c_tst)), verbose=false))

#@info("FD test grad_params_config with zygote")


@info("Energies test")

using JuLIP

emt = EMT()
at = bulk(:Cu, cubic=true) * 3
rattle!(at,0.6)

outer_nonL(x) = 2*x + cos(x)
e_me(θ) = outer_nonL(ACE.Models.ENERGY_me(standard, at, emt)(θ))

for ntest = 1:30
    c_tst = rand(length(BB),2) .- 0.5
    F = t ->  e_me(c_tst[:] + t)
    dF = t -> [collect(Iterators.flatten(Zygote.gradient(e_me,c_tst[:] + t)[1]))[i].val for i in 1:length(c_tst[:])]
    print_tf(@test fdtest(F, dF, zeros(length(c_tst)), verbose=false))
end
println()


# evalme = ACE.Models.EVAL_me(standard,cfg)
# test_fe(θ) = outer_nonL(evalme(θ))
# Zygote.gradient(e_me,c)[1]

#endl;