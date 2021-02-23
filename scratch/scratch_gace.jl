
using ACE

𝒓 = EuclideanVectorState("𝒓")
μ = SpeciesState("μ")
X = μ ⊗ 𝒓

X.rr
X.mu

@code_llvm X.rr

length(X)  # wait and see what we want `length` to mean...
