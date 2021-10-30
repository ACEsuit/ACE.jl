


##


using ACE, ACEbase
using Printf, Test, LinearAlgebra, ACE.Testing, Random
using ACE: evaluate, evaluate_d, SymmetricBasis, PIBasis, 
           grad_config, grad_params, O3
using ACEbase.Testing: fdtest

randconfig(B1p, nX) = ACEConfig( rand(PositionState{Float64}, B1p.bases[1], nX) )

##

@info("Basic test of LinearACEModel construction and evaluation")

# construct the 1p-basis
maxdeg = 6
ord = 3
Bsel = SimpleSparseBasis(ord, maxdeg) 

B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg)

# generate a configuration
cfg = randconfig(B1p, 10)

φ = ACE.Invariant()
basis = SymmetricBasis(φ, B1p, O3(), Bsel)

BB = evaluate(basis, cfg)
c = rand(length(BB)) .- 0.5
naive = ACE.LinearACEModel(basis, c, evaluator = :naive)
standard = ACE.LinearACEModel(basis, c, evaluator = :standard)


## FIO 

@info("Check FIO")
using ACEbase.Testing: test_fio 
println(@test(all(test_fio(naive; warntype = false))))
println(@test(all(test_fio(standard; warntype = false))))

##

evaluate(naive, cfg) ≈  evaluate(standard, cfg)
grad_params(naive, cfg) ≈ grad_params(standard, cfg)
grad_config(naive, cfg) ≈ grad_config(standard, cfg)

g1 = grad_config(naive, cfg)
g2 = grad_config(standard, cfg)

evaluate_ref(basis, cfg, c) = sum(evaluate(basis, cfg) .* c)

grad_config_ref(basis, cfg, c) = permutedims(evaluate_d(basis, cfg)) * c

grad_params_ref(basis, cfg, c) = evaluate(basis, cfg)

grad_params_config_ref(basis, cfg, c) = evaluate_d(basis, cfg)

(fun, funref, str) = (ACE.grad_params_config, grad_params_config_ref, "grad_params_config")

for (fun, funref, str) in [ 
         (evaluate, evaluate_ref, "evaluate"), 
         (ACE.grad_config, grad_config_ref, "grad_config"), 
         (ACE.grad_params, grad_params_ref, "grad_params"), 
         (ACE.grad_params_config, grad_params_config_ref, "grad_params_config"), 
      ]
   @info("Testing `$str` for different model evaluators")
   for ntest = 1:30
      local c, cfg
      cfg = rand(PositionState{Float64}, B1p.bases[1], nX) |> ACEConfig
      c = rand(length(basis)) .- 0.5 
      ACE.set_params!(naive, c)
      ACE.set_params!(standard, c)
      val = funref(basis, cfg, c)
      val_naive = fun(naive, cfg)
      val_standard = fun(standard, cfg)
      print_tf(@test( val ≈ val_naive ≈ val_standard ))
   end
   println()
end

##

@info("Test a Linear Model with EuclideanVector output")
maxdeg = 6; ord = 3
Bsel = SimpleSparseBasis(ord, maxdeg) 
B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg)
φ = ACE.EuclideanVector{Float64}()
basis = SymmetricBasis(φ, B1p, O3(), Bsel)

##
@info(" test evaluation of basis vs model ")

for ntest = 1:30 
   local cfg = randconfig(B1p, 10)
   local c = randn(length(BB)) ./ (1:length(BB)).^2
   BB = evaluate(basis, cfg)
   model = ACE.LinearACEModel(basis, c, evaluator = :standard)
   val1 = sum(c .* BB) 
   val2 = evaluate(model, cfg)
   print_tf(@test( val1 ≈ val2 ))
end

##
@info("test gradients of LinearACEModel")

c = randn(length(BB)) ./ (1:length(BB)).^2
BB = evaluate(basis, cfg)
model = ACE.LinearACEModel(basis, c, evaluator = :standard)

cfg = randconfig(B1p, 10)
evaluate(model, cfg)
grad_config(model, cfg)