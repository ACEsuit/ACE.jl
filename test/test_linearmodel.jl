


@testset "LinearACEModel"  begin

##


using ACE, ACEbase
using Printf, Test, LinearAlgebra, ACE.Testing, Random
using ACE: evaluate, evaluate_d, SymmetricBasis, NaiveTotalDegree, PIBasis
using ACEbase.Testing: fdtest
using ACE.Testing: __TestSVec

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


##

# evaluate(naivemodel, cfg)
# evaluate(standard, cfg)

evaluate_ref(basis, cfg, c) = sum(evaluate(basis, cfg) .* c)

grad_config_ref(basis, cfg, c) = 
      permutedims(evaluate_d(basis, cfg)) * c


for (fun, funref, str) in [ 
         (evaluate, evaluate_ref, "evaluate"), 
         (ACE.grad_config, grad_config_ref, "grad_config"), 
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




# println(@test(val_basis ≈ val_V))
# println(@test(evaluate(Vdag, Rs, Zs, z0) ≈ val_V))
# J = evaluate_d(basis, Rs, Zs, z0)
# grad_basis = real(sum(c[i] * J[i,:] for i = 1:length(c)))[:]
# grad_V = evaluate_d(V, Rs, Zs, z0)
# println(@test(grad_basis ≈ grad_V))
# println(@test(evaluate_d(Vdag, Rs, Zs, z0) ≈ grad_V))

# println(@test(all(JuLIP.Testing.test_fio(V))))

# ##

# # check multi-species
# maxdeg = 5
# Pr = transformed_jacobi(maxdeg, trans, rcut; pcut = 2)
# species = [:C, :O, :H]
# P1 = ACE.RnYlm1pBasis(Pr; species = [:C, :O, :H], D = D)
# basis = ACE.PIBasis(P1, 3, D, maxdeg)
# c = randcoeffs(basis)
# Vdag = combine(basis, c)
# V = standardevaluator(Vdag)
# Rs, Zs, z0 = ACE.rand_nhd(Nat, Pr, species)
# AA = evaluate(basis, Rs, Zs, z0)
# val_basis = real(sum(c .* evaluate(basis, Rs, Zs, z0)))
# val_V = evaluate(V, Rs, Zs, z0)
# println(@test(val_basis ≈ val_V))
# println(@test(evaluate(Vdag, Rs, Zs, z0) ≈ val_V))
# J = evaluate_d(basis, Rs, Zs, z0)
# grad_basis = real(sum(c[i] * J[i,:] for i = 1:length(c)))[:]
# grad_V = evaluate_d(V, Rs, Zs, z0)
# println(@test(grad_basis ≈ grad_V))
# println(@test(evaluate_d(Vdag, Rs, Zs, z0) ≈ grad_V))

# println(@test(all(JuLIP.Testing.test_fio(V))))

# ##

# @info("Check several properties of PIPotential")
# for species in (:X, :Si, [:C, :O, :H]), N = 1:5
#    local Rs, Zs, z0, V, Vdag, basis, P1, maxdeg, Nat, c, val_basis, val_V
#    maxdeg = 7
#    Nat = 15
#    P1 = ACE.RnYlm1pBasis(Pr; species = species)
#    basis = ACE.PIBasis(P1, N, D, maxdeg)
#    @info("species = $species; N = $N; length = $(length(basis))")
#    c = randcoeffs(basis)
#    Vdag = combine(basis, c)
#    V = standardevaluator(Vdag)
#    @info("check (de-)serialisation")
#    println(@test(all(JuLIP.Testing.test_fio(Vdag))))
#    @info("Check basis and potential match")
#    for ntest = 1:20
#       Rs, Zs, z0 = ACE.rand_nhd(Nat, Pr, species)
#       val_basis = real(sum(c .* evaluate(basis, Rs, Zs, z0)))
#       val_V = evaluate(V, Rs, Zs, z0)
#       print_tf(@test(val_basis ≈ val_V))
#    end
#    println()
#    @info("Check gradients")
#    for ntest = 1:20
#       Rs, Zs, z0 = ACE.rand_nhd(Nat, Pr, species)
#       V0 = evaluate(V, Rs, Zs, z0)
#       dV0 = evaluate_d(V, Rs, Zs, z0)
#       Us = [ rand(eltype(Rs)) .- 0.5 for _=1:length(Rs) ]
#       dV0_dUs = sum(transpose.(dV0) .* Us)
#       errs = []
#       for p = 2:12
#          h = 0.1^p
#          V_h = evaluate(V, Rs + h * Us, Zs, z0)
#          dV_h = (V_h - V0) / h
#          # @show norm(dAA_h - dAA_dUs, Inf)
#          push!(errs, norm(dV_h - dV0_dUs, Inf))
#       end
#       success = (/(extrema(errs)...) < 1e-3) || (minimum(errs) < 1e-10)
#       print_tf(@test success)
#    end
#    println()
#    @info("Check graph evaluator")
#    for ntest = 1:20
#       Rs, Zs, z0 = ACE.rand_nhd(Nat, Pr, species)
#       v = evaluate(V, Rs, Zs, z0)
#       vgr = evaluate(Vdag, Rs, Zs, z0)
#       print_tf(@test(v ≈ vgr))
#    end
#    println()
# end
# println()

# ##


# @info("Check Correctness of SHIP.PIPotential calculators")

# naive_energy(V::PIPotential, at) =
#       sum( evaluate(V, Rs, at.Z[j], at.Z[i])
#             for (i, j, Rs) in sites(at, cutoff(V)) )

# for N = 1:5
#    species = :Si
#    maxdeg = 7
#    Pr = transformed_jacobi(maxdeg, trans, 4.0; pcut = 2)
#    P1 = ACE.RnYlm1pBasis(Pr; species = species)
#    basis = ACE.PIBasis(P1, N, D, maxdeg)
#    @info("N = $N; length = $(length(basis))")
#    c = randcoeffs(basis)
#    V = combine(basis, c)
#    at = bulk(:Si) * (2,2,3)
#    rattle!(at, 0.1)
#    print("     energy: ")
#    println(@test energy(V, at) ≈ naive_energy(V, at) )
#    print("site-energy: ")
#    println(@test energy(V, at) ≈ sum( site_energy(V, at, n)
#                                           for n = 1:length(at) ) )
#    print("     forces: ")
#    println(@test JuLIP.Testing.fdtest(V, at; verbose=false))
#    print("site-forces: ")
#    println(@test JuLIP.Testing.fdtest( x -> site_energy(V, set_dofs!(at, x), 3),
#                                        x -> mat(site_energy_d(V, set_dofs!(at, x), 3))[:],
#                                        dofs(at); verbose=false ) )
# end


##
   
end
   