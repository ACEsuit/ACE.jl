
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------


using StaticArrays
import SymPy
using SymPy: symbols, simplify
using ACE.SphericalHarmonics: RSHBasis

# These are the expressions we use to test the real SH implementation
#  Y_l^m    =      1/√2 (Y_{lm} - i Y_{l,-m})
#  Y_l^{-m} = (-1)^m/√2 (Y_{lm} + i Y_{l,-m})
# cYt[i_p] = (1/sqrt(2)) * (rY[i_p] - im * rY[i_m])
# cYt[i_m] = (-1)^m * (1/sqrt(2)) * (rY[i_p] + im * rY[i_m])


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
   #   So Cj ≡ Y_{lj, |mj|} and Sj ≡ Y_{lj, -|mj|}
   #   and Y_{lj}^{±|mj|} can be expressed in terms of Cj, Sj.
   C = [ symbols("C$j", real=true) for j in 1:length(ll) ]
   S = [ symbols("S$j", real=true) for j in 1:length(ll) ]

   # now express the complex spherical harmonics in terms of the real ones
   # (note we don't need to refer to the cY list since it stores the same
   #  symbols as the C, S lists)
   cY = Vector{Any}(undef, n)
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
   expr = (real(coeff * prod(cY))).expand()
   verbose && println(expr)

   # next, we need to extract the prefactors
   # to get these we loop over all possible {C,S}1*{C,S}2*... combinations
   # below, CS is to conveniently access the relevant symbols
   # while signs is used to decide which symbol correspond to a +|m| or -|m|
   # basis function
   CS = [S'; C']
   signs = [- ones(Int, n)'; ones(Int, n)']
   real_basis = Any[]

   verbose && println("mm = $mm")
   for ii in CartesianIndices(ntuple(_ -> 2, n))
      ivec = SVector(ii.I...)
      term = prod(CS[ivec[α], α] for α = 1:n)
      pref = (expr.coeff(term)).subs(a, real(c)).subs(b, imag(c))
      if pref == 0; continue; end
      mm_i = [ signs[ivec[α], α] * abs(mm[α])  for α = 1:n ]
      verbose && println(term, " -> (", mm_i, ", ", pref, ")")
      push!(real_basis, (mm = mm_i, c = SymPy.N(pref)))
   end
   return [b for b in real_basis]
end


function convertc2r(cship::SHIP{T, NZ}) where {T, NZ}
   RSH = RSHBasis(cship.SH.maxL, T)
   alists =  ntuple(iz -> deepcopy(cship.alists[iz]), NZ)
   aalists = ntuple(iz -> deepcopy(cship.aalists[iz]), NZ)
   rcoeffs = ntuple(iz -> _convert_c2r_inner(cship.coeffs[iz],
                                             alists[iz],
                                             aalists[iz]), NZ )
   return RSHIP(cship.J, RSH, deepcopy(cship.zlist),
                alists, aalists, rcoeffs)
end

"""
This is the main conversion loop, which is called from the glue code
`convertc2r` and which itself calls the core function
`convert_c2r_1b` which converts a single basis function using SymPy.
"""
function _convert_c2r_inner(ccoeffs::AbstractVector{Complex{T}},
                            alist, aalist) where {T}
   rcoeffs = zeros(T, length(ccoeffs))
   missed = Any[]
   for iAA = 1:length(aalist)
      N = aalist.len[iAA]    # number of As to be multiplied
      # get the (z, k, l, m) infor the the terms in the product
      As = [ alist[aalist.i2Aidx[iAA, α]]  for α = 1:N ]
      izz = Int16[ As[α].z for α = 1:N ]
      kk  = IntS[ As[α].k for α = 1:N ]
      ll  = IntS[ As[α].l for α = 1:N ]
      mm  = IntS[ As[α].m for α = 1:N ]
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
         b_mm = IntS.(b.mm)
         # if the izklm key doesn't exist it means we have to add a
         # new AA-basis function to the aalist
         if !haskey(aalist.zklm2i, (izz, kk, ll, b_mm))
            push!(aalist, (izz, kk, ll, b_mm), alist)
            push!(rcoeffs, 0.0)
         end
         i_rAA = aalist.zklm2i[(izz, kk, ll, b_mm)]
         rcoeffs[i_rAA] += b.c
      end
   end
   if length(rcoeffs) > length(ccoeffs)
      @info("Added $(length(rcoeffs) - length(ccoeffs)) basis function(s).")
   end
   return rcoeffs
end
