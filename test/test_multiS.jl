
using LinearAlgebra: length
using ACE, ACEbase
using ACE: evaluate, SymmetricBasis, NaiveTotalDegree, PIBasis

using StaticArrays


##
    
# construct the 1p-basis
D = NaiveTotalDegree()
maxdeg = 6
ord = 3

B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, D = D)

# generate a configuration
nX = 54
Xs() = ACE.State(rr = rand(SVector{3, Float64}), u = rand())
cfg = ACEConfig([Xs() for i in 1:nX])
a(x) = fieldnames(typeof(x))

φ = ACE.Invariant()
pibasis = PIBasis(B1p, ord, maxdeg; property = φ)
basis = SymmetricBasis(pibasis, φ)
       
BB = evaluate(basis, cfg)

c1 = rand(length(BB))
c2 = rand(SVector{7,Float64}, length(BB))

singlSpecies = ACE.LinearACEModel(basis, c1, evaluator = :standard)
multiSpecies = ACE.LinearACEModel(basis, c2, evaluator = :standard)

@info("set_params!")
ACE.set_params!(singlSpecies,c1)
ACE.set_params!(multiSpecies,c2)

@info("evaluate")
@show evaluate(singlSpecies,cfg)
@show evaluate(multiSpecies,cfg)

@info("grad_params")
ACE.grad_params(singlSpecies,cfg)
ACE.grad_params(multiSpecies,cfg)

@info("grad_config")
ACE.grad_config(singlSpecies,cfg)
ACE.grad_config(multiSpecies,cfg)

@info("grad_config")
ACE.grad_params_config(singlSpecies,cfg)
ACE.grad_params_config(multiSpecies,cfg);
