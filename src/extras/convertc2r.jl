
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------

using StaticArrays
import SymPy
using SymPy: symbols, simplify
using PoSH.SphericalHarmonics: RSHBasis


"""
core of the C->R conversion, uses SymPy to express a product of
complex spherical harmonics as a product of real spherical harmonics, and
then computes the new coefficients.
"""
function convert_c2r_1b(ll, mm, c; verbose=false)
   # length of the tuple / length of the product
   n = length(ll)
   verbose && println("mm = $mm")

   # create symbols representing the real spherical harmonics
   # by analogy with exp(ikx) = cos(kx) + i sin(kx)
   C = [ symbols("C$j", real=true) for j in 1:length(ll) ]
   S = [ symbols("S$j", real=true) for j in 1:length(ll) ]

   # now express the complex spherical harmonics in terms of the real ones
   # (note we don't need to refer to the cY list since it stores the same
   #  symbols as the C, S lists)
   cY = Any[nothing for _=1:n]
   for j in 1:n
      if mm[j] == 0
         cY[j] = C[j]
      elseif mm[j] > 0
         cY[j] = (1/SymPy.sqrt(2)) * (C[j] - im * S[j])
      else
         cY[j] = ((-1)^mm[j]/SymPy.sqrt(2)) * (C[j] + im * S[j])
      end
    end

   # finally create the symbols for the prefactor
   a = symbols("a", real=true)
   b = symbols("b", real=true)
   coeff = a + im * b

   # let Sympy multiply and simplify the expression
   expr = simplify(real(coeff * prod(cY)))
   verbose && println(expr)

   # next, we need to extract the prefactors
   CS = [S; C]
   signs = Int[ - ones(n); ones(n) ]
   real_basis = Any[]

   verbose && println("mm = $mm")
   for ii in CartesianIndices(ntuple(_ -> length(CS), n))
      ivec = SVector(ii.I...)
      if length(unique(ivec)) != length(ivec); continue; end
      if sort(ivec) != ivec; continue; end
      term = prod(CS[ivec])
      pref = (expr.coeff(term)).subs(a, real(c)).subs(b, imag(c))
      if pref == 0; continue; end
      mm_i = signs[ivec] .* mm
      verbose && println(term, " -> (", mm_i, ", ", pref, ")")
      push!(real_basis, (mm = mm_i, c = SymPy.N(pref)))
   end
   return [b for b in real_basis]
end


function convertc2r(cship::SHIP{T, NZ}) where {T, NZ}
   RSH = RSHBasis(cship.SH.maxL, T)
   rcoeffs = ntuple( iz -> _convert_c2r_inner(cship.coeffs[iz],
                                              cship.alists[iz],
                                              cship.aalists[iz]), NZ )
   return RSHIP(cship.J, RSH, cship.zlist,
                cship.alists, cship.aalists, rcoeffs)
end

"""
This is the main conversion loop, which is called from the glue code
`convertc2r` and which itself calls the core function
`convert_c2r_1b` which converts a single basis function using SymPy.
"""
function _convert_c2r_inner(ccoeffs::AbstractVector{Complex{T}},
                            alist, aalist) where {T}
   rcoeffs = zeros(T, length(ccoeffs))
   for iAA = 1:length(aalist)
      N = aalist.len[iAA]    # number of As to be multiplied
      # get the (z, k, l, m) infor the the terms in the product
      As = [ alist[aalist.i2Aidx[iAA, α]]  for α = 1:N ]
      izz = [ As[α].z for α = 1:N ]
      kk = [ As[α].k for α = 1:N ]
      ll = [ As[α].l for α = 1:N ]
      mm = [ As[α].m for α = 1:N ]
      # now we convert the ∏ Y_{lα}^{mα} to several ∏ Y_{lα, mα},
      # i.e. we get a LIST of several real basis functions
      rbasis = convert_c2r_1b(ll, mm, ccoeffs[iAA])
      # so now we need to loop through the rbasis and add the new
      # coefficients to rcoeffs as specified by  the basis functions
      for b in rbasis
         # b -> b.mm, b.c -> (ll, b.mm) define a product of real SHs
         # and b.c is the new coefficient; but we first need to identify
         # the correct index in `rcoeffs` which is the same as the
         # corresponding index in `aalist`
         i_rAA = aalist.zklm2i[(izz, kk, ll, b.mm)]
         rcoeffs[i_rAA] += b.c
      end
   end
   return rcoeffs
end


# # a basis function is defined by an l and an m tuple:
# ll = [1, 3, 3, 4]
# mm = [1, -2, -1, 1]
# c = rand() + im * rand()
#
# convert_c2r_1b(ll, mm, c)
