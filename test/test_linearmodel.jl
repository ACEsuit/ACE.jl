
@testset "LinearACEModel"  begin

##


using ACE, ACEbase
using Printf, Test, LinearAlgebra, ACE.Testing, Random
using ACE: evaluate, evaluate_d, SymmetricBasis, NaiveTotalDegree, PIBasis
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
Xs = rand(EuclideanVectorState, B1p.bases[1], nX)
cfg = ACEConfig(Xs)

φ = ACE.Invariant()
pibasis = PIBasis(B1p, ord, maxdeg; property = φ)
basis = SymmetricBasis(pibasis, φ)

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


# evaluate(naivemodel, cfg)
# evaluate(standard, cfg)
# ACE.grad_config(standard, cfg)

evaluate_ref(basis, cfg, c) = sum(evaluate(basis, cfg) .* c)

grad_config_ref(basis, cfg, c) = permutedims(evaluate_d(basis, cfg)) * c

grad_params_ref(basis, cfg, c) = evaluate(basis, cfg)

grad_params_config_ref(basis, cfg, c) = evaluate_d(basis, cfg)


for (fun, funref, str) in [ 
         (evaluate, evaluate_ref, "evaluate"), 
         (ACE.grad_config, grad_config_ref, "grad_config"), 
         (ACE.grad_params, grad_params_ref, "grad_params"), 
         (ACE.grad_params_config, grad_params_config_ref, "grad_params_config"), 
      ]
   @info("Testing `$str` for different model evaluators")
   for ntest = 1:30
      cgf = rand(EuclideanVectorState, B1p.bases[1], nX) |> ACEConfig
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

@info("Check linear model gradients for a covariant vector")
φ = ACE.EuclideanVector{ComplexF64}()
pibasis = PIBasis(B1p, ord, maxdeg; property = φ, isreal=false)
basis = SymmetricBasis(pibasis, φ; isreal=false)
c = rand(length(basis)) .- 0.5
standard = ACE.LinearACEModel(basis, c, evaluator = :standard)
evaluate(standard, cfg)
ACE.grad_config(standard, cfg)

for ntest = 1:30
   Us = rand(SVector{3, Float64}, length(cfg))
   a = rand(3)
   _cfgt = t -> ACEConfig( cfg.Xs + t * Us )
   _cfgt(0.1)
   F = t -> dot(a, real(evaluate(standard, _cfgt(t)).val))
   dF = t -> dot(a, sum(real.(d) * u 
               for (d, u) in zip( ACE.grad_config(standard, _cfgt(t)), Us ) ) )
   print_tf(@test( all( ACEbase.Testing.fdtest(F, dF, 0.0, verbose=false) )))
end 

##
   
end
   