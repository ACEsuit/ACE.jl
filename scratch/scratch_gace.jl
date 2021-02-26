using ACE
using ACE: alloc_temp, alloc_B, AtomState

species = [:Ti, :Mo]
Bμ0 = ACE.Species1PBasisCtr(species)
Bμ1 = ACE.Species1PBasisNeig(species)
Rn = ACE.Rn1pBasis(ACE.Utils.radial_basis())
Ylm = ACE.Ylm1pBasis(10)

B1p = ACE.Product1PBasis( (Bμ0, Bμ1, Rn, Ylm) )

ACE.symbols(B1p)
ACE.indexrange(B1p)

# TODO : automate this...
ACE.init1pspec!(B1p, maxdeg=5)
length(B1p)

ace = PIBasis(B1p, ACE.One1pBasis(), 3, 5)



function rand_state()
   𝒓 = ACE.rand_radial(Rn.R) * ACE.Random.rand_sphere()
   μ = rand(species)
   return AtomState(μ, 𝒓)
end

A = alloc_B(B1p)
tmp = alloc_temp(B1p)

Xs = [ rand_state() for _ = 1:30 ]
X0 = rand_state()

ACE.evaluate!(A, tmp, B1p, Xs, X0)
using BenchmarkTools
@btime ACE.evaluate!($A, $tmp, $B1p, $Xs, $X0)


AA = alloc_B(ace)
tmpace = alloc_temp(ace)
ACE.evaluate!(AA, tmpace, ace, Xs, X0)
