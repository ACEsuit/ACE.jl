using ACE
using ACE: alloc_temp, alloc_B, AtomState

species = [:Ti, :Mo]
Bμ0 = ACE.Species1PBasisCtr(species)
Bμ1 = ACE.Species1PBasisNeig(species)
Rn = ACE.Rn1pBasis(ACE.Utils.radial_basis())
Ylm = ACE.Ylm1pBasis(10)
B1p = ACE.Product1pBasis( (Bμ0, Bμ1, Rn, Ylm) )

ACE.symbols(B1p)
ACE.indexrange(B1p)

# TODO : automate this...
ACE.init1pspec!(B1p, maxdeg=5)
length(B1p)

φ = ACE.Invariant()
ace = PIBasis(B1p, ACE.One1pBasis(), 3, 5) |> length #  length ~ 62,000
ace = PIBasis(B1p, ACE.One1pBasis(), 3, 5; property = φ)   # filtered: length ~ 7,000  !!!

# full specification of the 𝑨 Basis
ACE.get_spec(ace)

symace = ACE.SymmetricBasis(ace, φ)


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



B = alloc_B(symace)
tmp = alloc_temp(symace)
ACE.evaluate!(B, tmp, symace, Xs, X0)


#---

using StaticArrays, Random

function rand_rot(Xs::AbstractVector{<: AtomState})
   K = (@SMatrix rand(3,3)) .- 0.5
   Q = exp(K - K')
   return [ AtomState(X.mu, Q * X.rr) for X in Xs ]
end

rand_refl(Xs::AbstractVector{<: AtomState}) = (σ = rand([-1, 1]);
      [ AtomState(X.mu, σ * X.rr)  for X in Xs ] )

rand_sym(Xs) = shuffle(rand_refl(rand_rot(Xs)))



for ntest = 1:10
   Xs1 = rand_sym(Xs)
   B1 = alloc_B(symace)
   B1 = ACE.evaluate!(B1, tmp, symace, Xs1, X0)
   @show norm(B1 - B)
   # isapprox(B, B1; rtol = 1e-10)  # use this intead of ≈
end
