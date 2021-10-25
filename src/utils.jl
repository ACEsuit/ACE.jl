

module Utils

import ACE

import ACE: PolyTransform, transformed_jacobi, Rn1pBasis,
            init1pspec!, Ylm1pBasis,
            Product1pBasis, SimpleSparseBasis

# import ACE.PairPotentials: PolyPairBasis

# - simple ways to construct a radial basis
# - construct a descriptor
# - simple wrappers to generate RPI basis functions (ACE + relatives)

@doc raw"""
`Rn_basis` construct a ``R_n`` basis; all arguments are keyword arguments with
defaults:
* `r0 = 1.0`
* `trans = PolyTransform(2, r0)`
* `D = NaiveTotalDegree()`
* `maxdeg = 6`
* `rcut = 2.5`
* `rin = 0.5 * r0`
* `pcut = 2`
* `pin = 0`
* `constants = false`
"""
function Rn_basis(;
      # transform parameters
      r0 = 1.0,
      trans = PolyTransform(2, r0),
      # degree parameters
      maxdeg = 6,
      # radial basis parameters
      rcut = 2.5,
      rin = 0.5 * r0,
      pcut = 2,
      pin = 0,
      constants = false, 
      varsym = :rr, 
      nsym = :n)

   J = transformed_jacobi(maxdeg, trans, rcut, rin; pcut=pcut, pin=pin)
   return Rn1pBasis(J, varsym = varsym, nsym = nsym)
end

@doc raw"""
Construct a ``R_n * Y_l^m`` 1-particle basis.
All arguments are keyword arguments; see documentation of `ACE.Utils.Rn_basis`.
"""
function RnYlm_1pbasis(; maxdeg=6, maxL = maxdeg, varsym = :rr, idxsyms = (:n, :l, :m), 
                         Bsel = nothing, kwargs...)
   Rn = Rn_basis(; maxdeg = maxdeg, varsym = varsym, nsym = idxsyms[1],
                   kwargs...)
   Ylm = Ylm1pBasis(maxL, varsym = varsym, lsym = idxsyms[2], msym = idxsyms[3])
   B1p = ACE.Product1pBasis((Rn, Ylm))
   if Bsel != nothing 
      init1pspec!(B1p, Bsel)
   end
   return B1p
end

function BondBasisSelector(Bsel::ACE.SparseBasis; isym=:be, bond_weight = 1.0, env_weight = 1.0)
   weight_cat = Dict(:bond => 1.0, :env=> 1.0) 
   BondSelector = ACE.CategorySparseBasis(Bsel.maxorder, isym, [:bond, :env];
   p = Bsel.p, 
   weight = Bsel.weight, 
   maxdegs = Bsel.maxdegs,
   minorder_dict = Dict( :bond => 1),
   maxorder_dict = Dict( :bond => 1),
   weight_cat = Dict(:bond => bond_weight, :env=> env_weight) 
   )
   return BondSelector
end

function SymmetricBond_basis(ϕ::ACE.AbstractProperty, env::ACE.BondEnvelope, Bsel::ACE.SparseBasis; RnYlm = nothing, kwargs...)
   BondSelector =  BondBasisSelector(Bsel; kwargs...)
   if RnYlm == nothing
       RnYlm = RnYlm_1pbasis(;   r0 = ACE.cutoff_radialbasis(env), 
                                           rin = 0.0,
                                           trans = PolyTransform(2, ACE.cutoff_radialbasis(env)), 
                                           pcut = 2,
                                           pin = 0, 
                                           kwargs...
                                       )
   end
   Bc = ACE.Categorical1pBasis([:bond, :env]; varsym = :be, idxsym = :be )
   B1p =  Bc * RnYlm * env
   return ACE.SymmetricBasis(ϕ, B1p, BondSelector)
end

# invariant_basis(; kwargs...) =
#       symm_basis(ACE.Invariant(); kwargs...)

# symm_basis(φ; maxν = 3, maxdeg = 6, kwargs...) =
#       ACE.SymmetricBasis(φ,
#                          RnYlm_1pbasis(; maxdeg=maxdeg, kwargs...),
#                          maxν,
#                          maxdeg)

# function rpi_basis(; species = :X, N = 3,
#       # transform parameters
#       r0 = 2.5,
#       trans = PolyTransform(2, r0),
#       # degree parameters
#       D = SparsePSHDegree(),
#       maxdeg = 8,
#       # radial basis parameters
#       rcut = 5.0,
#       rin = 0.5 * r0,
#       pcut = 2,
#       pin = 0,
#       constants = false,
#       rbasis = transformed_jacobi(get_maxn(D, maxdeg, species), trans, rcut, rin;
#                                   pcut=pcut, pin=pin),
#       # one-particle basis
#       Basis1p = RnYlm1pBasis,
#       basis1p = Basis1p(rbasis; species = species, D = D) )
#
#    return RPIBasis(basis1p, N, D, maxdeg, constants)
# end
#
# descriptor = rpi_basis
# ace_basis = rpi_basis
#
# function pair_basis(; species = :X,
#       # transform parameters
#       r0 = 2.5,
#       trans = PolyTransform(2, r0),
#       # degree parameters
#       maxdeg = 8,
#       # radial basis parameters
#       rcut = 5.0,
#       rin = 0.5 * r0,
#       pcut = 2,
#       pin = 0,
#       rbasis = transformed_jacobi(maxdeg, trans, rcut, rin; pcut=pcut, pin=pin))
#
#    return PolyPairBasis(rbasis, species)
# end




end
