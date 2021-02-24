
using ACE
using ACE: alloc_temp, alloc_B

X = AtomState()

species = [:Ti, :Mo]
Bμ0 = ACE.Species1PBasisCtr(species)
Bμ1 = ACE.Species1PBasisNeig(species)
# RnYlm = ACE.Utils.RnYlm_basis(species = species
#    )

Rn = ACE.Rn1pBasis(ACE.Utils.radial_basis())
Ylm = ACE.Ylm1pBasis(10)

B1p = ACE.Product1PBasis( (Bμ0, Bμ1, Rn, Ylm) )

function rand_state()
   𝒓 = ACE.rand_radial(Rn.R) * ACE.Random.rand_sphere()
   μ = rand(species)
   return AtomState(μ, 𝒓)
end

# generate a random specification
spec = [ (rand(1:2), rand(1:2), rand(1:10), rand(1:10)) for _=1:10 ]
append!(B1p.spec, spec)

tmp = alloc_temp(B1p)
A = zeros(ComplexF64, length(B1p))
alloc_B(B1p)
Xs = [ rand_state() for _ = 1:10 ]
X0 = rand_state()

ACE.evaluate!(A, tmp, B1p, Xs, X0)
