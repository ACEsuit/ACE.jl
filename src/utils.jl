

module Utils

import ACE

import ACE: PolyTransform, transformed_jacobi, Rn1pBasis,
            init1pspec!, Ylm1pBasis,
            Product1pBasis, SimpleSparseBasis

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
   return Rn1pBasis(J; varsym = varsym, nsym = nsym)
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



end
