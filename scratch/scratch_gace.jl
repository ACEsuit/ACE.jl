
using ACE
using ACE: alloc_temp, alloc_B

X = AtomState()

species = [:Ti, :Mo]
Bμ0 = ACE.Species1PBasisCtr(species)
Bμ1 = ACE.Species1PBasisNeig(species)
RnYlm = ACE.Utils.RnYlm_basis(species = species)

B1p = ACE.Product1PBasis( (Bμ0, Bμ1, RnYlm) )

# generate a random specification
spec = [ (rand(1:2), rand(1:2), rand(1:10)) for _=1:10 ]
append!(B1p.spec, spec)

tmp = alloc_temp(B1p)
A = zeros(ComplexF64, length(B1p))

Xs = [ rand(RnYlm) for _ = 1:10 ]
X0 = rand(RnYlm)

ACE.evaluate!(A, tmp, B1p, Xs, X0)
