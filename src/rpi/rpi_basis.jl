
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


using SparseArrays: SparseMatrixCSC, sparse, sprand
using LinearAlgebra: mul! 

"""
`struct RPIBasis`
"""
struct RPIBasis{T, BOP, NZ} <: IPBasis
   pibasis::PIBasis{BOP, NZ}
   A2Bmaps::NTuple{NZ, SparseMatrixCSC{T, Int}}
end

Base.length(basis::RPIBasis, iz0::Integer) = size(basis.A2Bmaps[iz0], 1)

Base.length(basis::RPIBasis) = sum(length(basis, iz0)
                                    for iz0 = 1:numz(basis.pibasis))

Base.eltype(::RPIBasis{T}) where {T}  = T


# ------------------------------------------------------------------------
#    Basis construction code
# ------------------------------------------------------------------------


function RPIBasis(basis1p::OneParticleBasis, N::Integer,
                  D::AbstractDegree, maxdeg::Real)
   # construct a permutation-invariant basis
   # TODO: filter out the even ll basis functions!
   pibasis = PIBasis(basis1p, N, D, maxdeg; filter = _rpi_filter)

   # construct the cg matrices
   rotc = Rot3DCoeffs()
   A2Bmaps = ntuple(iz0 -> _rpi_A2B_matrix(rotc, pibasis, iz0), numz(pibasis))
   # A2Bmaps = ntuple(iz0 -> sprand(10,10, 0.1), numz(pibasis))

   return RPIBasis(pibasis, A2Bmaps)
end

_rpi_filter(pib::PIBasisFcn{0}) = false
_rpi_filter(pib::PIBasisFcn{1}) = (pib.oneps[1].l == 0)
_rpi_filter(pib::PIBasisFcn) = iseven( sum(b.l for b in pib.oneps) )

function _rpi_A2B_matrix(rotc::Rot3DCoeffs,
                         pibasis::PIBasis,
                         iz0)
   # allocate triplet format
   Irow, Jcol, vals = Int[], Int[], eltype(pibasis.basis1p)[]
   # count the number of PI basis functions = number of rows
   idxB = 0
   # loop through all (zz, kk, ll) tuples; each specifies 1 to several B
   for i = 1:length(pibasis.inner[iz0])
      # get the specification of the ith basis function
      pib = get_basis_spec(pibasis, iz0, i)
      # skip it unless all m are zero, because we want to consider each
      # (nn, ll) block only once.
      if !all(b.m == 0 for b in pib.oneps)
         continue
      end
      # get the rotation-coefficients for this basis group
      # the bs are the basis functions corresponding to the columns
      U, bcols = _rpi_coupling_coeffs(pibasis, rotc, pib)
      # loop over the rows of Ull -> each specifies a basis function
      for irow = 1:size(U, 1)
         idxB += 1
         # loop over the columns of U / over brows
         for (icol, bcol) in enumerate(bcols)
            # this is a subtle step: bcol and bcol_ordered are equivalent
            # permutation-invariant basis functions. This means we will
            # add the same PI basis function several times, but in the call to
            # `sparse` the values will just be added.
            bcol_ordered = _get_ordered(pibasis, bcol)
            idxAA = pibasis.inner[iz0].b2iAA[bcol_ordered]
            push!(Irow, idxB)
            push!(Jcol, idxAA)
            push!(vals, U[irow, icol])
         end
      end
   end
   # create CSC: [   triplet    ]  nrows   ncols
   return sparse(Irow, Jcol, vals, idxB, length(pibasis.inner[iz0]))
end


# U, bcols = rpi_coupling_coeffs(rotc, pib)

"""
this is essentially a wrapper function around Rotations3D.rpi_basis,
and is just meant to translate between different representations
"""
function _rpi_coupling_coeffs(pibasis, rotc::Rot3DCoeffs, pib::PIBasisFcn{N}
                              ) where {N}
   # convert to zz, ll, nn tuples
   zz, nn, ll, _ = _b2znlms(pib)
   # construct the RPI coupling coefficients
   U, Ms = Rotations3D.rpi_basis(rotc, zz, nn, ll)
   # convert the Ms into basis functions
   rpibs = [ _znlms2b(zz, nn, ll, mm, pib.z0) for mm in Ms ]
   return U, rpibs
end

_b2znlms(pib::PIBasisFcn{N}) where {N} = (
   SVector(ntuple(n -> pib.oneps[n].z, N)...),
   SVector(ntuple(n -> pib.oneps[n].n, N)...),
   SVector(ntuple(n -> pib.oneps[n].l, N)...),
   SVector(ntuple(n -> pib.oneps[n].m, N)...) )

_znlms2b(zz, nn, ll, mm = zero(ll), z0 = AtomicNumber(0)) =
   PIBasisFcn( z0, ntuple(i -> PSH1pBasisFcn(nn[i], ll[i], mm[i], zz[i]),
                          length(zz)) )

function _get_ordered(pibasis, pib::PIBasisFcn{N}) where {N}
   inner = pibasis.inner[z2i(pibasis, pib.z0)]
   iAs = [ inner.b2iA[b] for b in pib.oneps ]
   p = sortperm(iAs)
   return PIBasisFcn(pib.z0, ntuple(i -> pib.oneps[p[i]], N))
end

# ------------------------------------------------------------------------
#    Evaluation code
# ------------------------------------------------------------------------

alloc_temp(basis::RPIBasis, args...) =
   ( AA = alloc_B(basis.pibasis, args...),
     tmp_pibasis = alloc_temp(basis.pibasis, args...)
   )

function evaluate!(B, tmp, basis::RPIBasis, Rs, Zs, z0)
   AA = evaluate!(tmp.AA, tmp.tmp_pibasis, basis.pibasis, Rs, Zs, z0)
   mul!(B, basis.A2Bmaps[z2i(basis.pibasis, z0)], AA)
   return B
end
