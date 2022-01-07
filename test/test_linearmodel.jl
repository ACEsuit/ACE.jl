


##


using ACE, ACEbase, StaticArrays
using Printf, Test, LinearAlgebra, ACE.Testing, Random
using ACE: evaluate, evaluate_d, SymmetricBasis, PIBasis, 
           grad_config, grad_params, O3
using ACEbase.Testing: fdtest, println_slim 


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
Xs = cfg.Xs

φ = ACE.Invariant()
basis = SymmetricBasis(φ, B1p, O3(), Bsel)

BB = evaluate(basis, cfg)
c = rand(length(BB)) .- 0.5
naive = ACE.LinearACEModel(basis, c, evaluator = :naive)
standard = ACE.LinearACEModel(basis, c, evaluator = :standard)

## FIO 

@info("Check FIO")
using ACEbase.Testing: test_fio 
println_slim(@test(all(test_fio(naive; warntype = false))))
println_slim(@test(all(test_fio(standard; warntype = false))))

##


evaluate_ref(basis, cfg, c) = sum(evaluate(basis, cfg) .* c)
grad_config_ref(basis, cfg, c) = permutedims(evaluate_d(basis, cfg)) * c
grad_params_ref(basis, cfg, c) = evaluate(basis, cfg)
grad_params_config_ref(basis, cfg, c) = evaluate_d(basis, cfg)

evaluate(naive, cfg) ≈  evaluate(standard, cfg)
evaluate_ref(basis, cfg, c) ≈ evaluate(naive, cfg)
grad_params(naive, cfg) ≈ grad_params(standard, cfg)
grad_config(naive, cfg) ≈ grad_config(standard, cfg)


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
      cfg = randconfig(B1p, 10)
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

@info("Evaluate LinearACEModel with vector")
for _ = 1:20
   cfg = randconfig(B1p, rand(8:15))
   print_tf(@test evaluate(standard, cfg) ≈ evaluate(standard, cfg.Xs))
   print_tf(@test grad_config(standard, cfg) ≈ grad_config(standard, cfg.Xs))
   print_tf(@test grad_params(standard, cfg) ≈ grad_params(standard, cfg.Xs))
end
println() 

##

@info("Test a Linear Model with EuclideanVector output")
maxdeg = 6; ord = 3
Bsel = SimpleSparseBasis(ord, maxdeg) 
B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg)
φ = ACE.EuclideanVector{Float64}()
basis = SymmetricBasis(φ, B1p, O3(), Bsel; isreal=true)

##
@info(" test evaluation of basis vs model ")

for ntest = 1:30 
   local cfg, c  
   cfg = randconfig(B1p, 10)
   BB = evaluate(basis, cfg)
   c = randn(length(BB)) ./ (1:length(BB)).^2
   model = ACE.LinearACEModel(basis, c, evaluator = :standard)
   val1 = sum(c .* BB) 
   val2 = evaluate(model, cfg)
   print_tf(@test( val1 ≈ val2 ))
end
println() 

##
@info("test gradients of LinearACEModel of equivariant vector")

BB = evaluate(basis, cfg)
c = randn(length(BB)) ./ (1:length(BB)).^2
model = ACE.LinearACEModel(basis, c, evaluator = :standard)

cfg = randconfig(B1p, 10)
evaluate(model, cfg)

dBB = evaluate_d(basis, cfg)
val1 = sum(c[i] * dBB[i,:] for i = 1:length(basis))
val2 = grad_config(model, cfg)
println_slim( @test( val1 ≈ val2 ))

##

@info("And a finite-difference test for good measure")

for ntest = 1:30 
   w = randn(3)
   Us = randn(SVector{3, Float64}, length(cfg))
   cfg.Xs + 0.01 * Us
   F = t -> real(w' * evaluate(model, ACEConfig(cfg.Xs + t * Us)).val)
   F(0.01)
   dF = t -> sum( real(w' * gi.rr * Ui) for (Ui, gi) in zip(Us, 
                           grad_config(model, ACEConfig(cfg.Xs + t * Us))) )
   dF(0.01)
   print_tf(@test all(ACEbase.Testing.fdtest(F, dF, 0.0, verbose=false)))
end
println() 