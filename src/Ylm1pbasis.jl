
# --------------------------------------------------------------------------
# ACE.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------

import ACE.SphericalHarmonics: SHBasis, index_y


@doc raw"""
`struct Ylm1pBasis <: OneParticleBasis`

One-particle basis of the form
```math
\phi_{lm}({\bm r}) = Y_l^m(\hat{\br r})
```
Fundamental building block of ACE basis sets of the form
```math
   R_{nl}^{\mu_i, \mu_j}(r_{ij}) Y_l^m(\hat{\bm r})
```
This type basically just translates the `SHBasis` into a valid one-particle
basis.
"""
mutable struct Ylm1pBasis{T} <: OneParticleBasis{Complex{T}}
   SH::SHBasis{T}  # SH = Ylm
end

# TODO: allow for the possibility that the symbol where the
#       position is stored is not `rr` but something else!
#
# TODO: Should we drop this type altogether and just replace it with SHBasis?


# ---------------------- Implementation of Ylm1pBasis


Ylm1pBasis(maxL::Integer, T = Float64) = Ylm1pBasis(SHBasis(maxL, T))

Base.length(basis::Ylm1pBasis) = length(basis.SH)

# -> TODO : figure out how to do this well!!!
# Base.rand(basis::Ylm1pBasis) =
#       AtomState(rand(basis.zlist.list), ACE.Random.rand_vec(basis.J))

function get_spec(basis::Ylm1pBasis)
   spec = NamedTuple{(:n, :l), (Int, Int)}[]
   for l = 0:basis.SH.maxL, m = -l:l
      spec[index_y(l, m)] = Ylm1pBasisFcn(l, m)
   end
   return spec
end

==(P1::Ylm1pBasis, P2::Ylm1pBasis) =  ACE._allfieldsequal(P1, P2)

write_dict(basis::Ylm1pBasis{T}) where {T} = Dict(
      "__id__" => "ACE_Ylm1pBasis",
          "SH" => write_dict(basis.SH) )

read_dict(::Val{:ACE_Ylm1pBasis}, D::Dict) = Ylm1pBasis(read_dict(D["SH"]))

fltype(basis::Ylm1pBasis{T}) where T = Complex{T}
rfltype(basis::Ylm1pBasis{T}) where T = T

symbols(::Ylm1pBasis) = [:l, :m]


function indexrange(basis::Ylm1pBasis)
   maxl = basis.SH.maxL
   # note we create a stupid tensor product domain and then make up for it
   # by using an index filter during the basis generation process
   return Dict( :l => 0:maxl, :m => -maxl:maxl )
end

isadmissible(b, basis::Ylm1pBasis) =
      (  (0 <= b.l <= basis.SH.maxL) &&
         (-b.l <= b.m <= b.l) )

# ---------------------------  Evaluation code
#

alloc_B(basis::Ylm1pBasis) = alloc_B(basis.SH)

alloc_temp(basis::Ylm1pBasis) = alloc_temp(basis.SH)

evaluate!(B, tmp, basis::Ylm1pBasis, X::AbstractState) =
      evaluate!(B, tmp, basis.SH, X.rr)

degree(b, basis::Ylm1pBasis) = b.l

get_index(basis::Ylm1pBasis, b) = index_y(b.l, b.m)

# alloc_temp_d(basis::RnYlm1pBasis, args...) =
#       (
#         BJ = alloc_B(basis.J, args...),
#         tmpJ = alloc_temp(basis.J, args...),
#         BY = alloc_B(basis.SH, args...),
#         tmpY = alloc_temp(basis.SH, args...),
#         dBJ = alloc_dB(basis.J, args...),
#         tmpdJ = alloc_temp_d(basis.J, args...),
#         dBY = alloc_dB(basis.SH, args...),
#         tmpdY = alloc_temp_d(basis.SH, args...),
#        )
#
# function add_into_A_dA!(A, dA, tmpd, basis::RnYlm1pBasis, R, iz::Integer, iz0::Integer)
#    r = norm(R)
#    R̂ = R / r
#    # evaluate the r-basis and the R̂-basis for the current neighbour at R
#    evaluate_d!(tmpd.BJ, tmpd.dBJ, tmpd.tmpdJ, basis.J, r)
#    evaluate_d!(tmpd.BY, tmpd.dBY, tmpd.tmpdY, basis.SH, R)
#    # add the contributions to the A_zklm, ∇A
#    @inbounds for (i, nlm) in enumerate(basis.spec)
#       iY = index_y(nlm.l, nlm.m)
#       A[i] += tmpd.BJ[nlm.n] * tmpd.BY[iY]
#       dA[i] = (tmpd.dBJ[nlm.n] * tmpd.BY[iY]) * R̂ + tmpd.BJ[nlm.n] * tmpd.dBY[iY]
#    end
#    return nothing
# end
