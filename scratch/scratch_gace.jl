
using ACE

𝒓 = PositionState("𝒓")
μ = SpeciesState("μ")
X = μ ⊗ 𝒓

length(X)  # wait and see what we want `length` to mean...
