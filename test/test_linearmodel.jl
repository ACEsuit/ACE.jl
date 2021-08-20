


##


using ACE, ACEbase
using Printf, Test, LinearAlgebra, ACE.Testing, Random
using ACE: evaluate, evaluate_d, SymmetricBasis, NaiveTotalDegree, PIBasis, 
           grad_config, grad_params, O3
using ACEbase.Testing: fdtest

##

@info("Basic test of LinearACEModel construction and evaluation")

# construct the 1p-basis
D = NaiveTotalDegree()
maxdeg = 6
ord = 3

B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, D = D)

# generate a configuration
nX = 10
Xs = rand(PositionState{Float64}, B1p.bases[1], nX)
cfg = ACEConfig(Xs)

φ = ACE.Invariant()
pibasis = PIBasis(B1p, O3(), ord, maxdeg; property = φ)
basis = SymmetricBasis(φ, O3(), pibasis)

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
grad_params(naive, cfg) ≈  grad_params(standard, cfg)
grad_config(naive, cfg) ≈ grad_config(standard, cfg)

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
