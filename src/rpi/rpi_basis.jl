
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


using SparseArrays: SparseMatrixCSC, sparse
using LinearAlgebra: mul!

"""
`struct RPIBasis`
"""
struct RPIBasis{T, BOP, NZ} <: IPBasis
   pibasis::PIBasis{BOP, NZ}
   A2Bmaps::NTuple{NZ, SparseMatrixCSC{T, Int}}
   Bz0inds::NTuple{NZ, UnitRange{Int}}
end

Base.length(basis::RPIBasis, iz0::Integer) = size(basis.A2Bmaps[iz0], 1)

Base.length(basis::RPIBasis) = sum(length(basis, iz0)
                                    for iz0 = 1:numz(basis.pibasis))

Base.eltype(::RPIBasis{T}) where {T}  = T

numz(basis::RPIBasis) = numz(basis.pibasis)
i2z(basis::RPIBasis, i::Integer) = i2z(basis.pibasis, i)
z2i(basis::RPIBasis, z::AtomicNumber) = z2i(basis.pibasis, z)

cutoff(basis::RPIBasis) = cutoff(basis.pibasis)

# ------------------------------------------------------------------------
#    FIO code
# ------------------------------------------------------------------------

==(B1::RPIBasis, B2::RPIBasis) = (B1.pibasis == B2.pibasis)

write_dict(basis::RPIBasis) = Dict(
      "__id__" => "SHIPs_RPIBasis",
      "pibasis" => write_dict(basis.pibasis),
   )

read_dict(::Val{:SHIPs_RPIBasis}, D::Dict) =
   RPIBasis(read_dict(D["pibasis"]))

# ------------------------------------------------------------------------
#    Basis construction code
# ------------------------------------------------------------------------


RPIBasis(basis1p::OneParticleBasis, N::Integer,
         D::AbstractDegree, maxdeg::Real) =
   RPIBasis(PIBasis(basis1p, N, D, maxdeg; filter = _rpi_filter))

function RPIBasis(pibasis::PIBasis)
   basis1p = pibasis.basis1p

   # construct the cg matrices
   rotc = Rot3DCoeffs()
   A2Bmaps = ntuple(iz0 -> _rpi_A2B_matrix(rotc, pibasis, iz0), numz(pibasis))

   # construct the indices within the B vector to which the A2Bmaps map.
   Bz0inds = UnitRange{Int}[]
   idx0 = 0
   for i = 1:length(A2Bmaps)
      len = size(A2Bmaps[i], 1)
      push!(Bz0inds, (idx0+1):(idx0+len))
      idx0 += len
   end

   return RPIBasis(pibasis, A2Bmaps, tuple(Bz0inds...))
end

_rpi_filter(pib::PIBasisFcn{0}) = false
_rpi_filter(pib::PIBasisFcn{1}) = (pib.oneps[1].l == 0)
_rpi_filter(pib::PIBasisFcn) = (
      iseven( sum(b.l for b in pib.oneps) ) &&
      (sum(b.m for b in pib.oneps) == 0) )

function _rpi_A2B_matrix(rotc::Rot3DCoeffs,
                         pibasis::PIBasis,
                         iz0)
   # allocate triplet format
   Irow, Jcol, vals = Int[], Int[], real(eltype(pibasis.basis1p))[]
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
            bcol_ordered = SHIPs._get_ordered(pibasis, bcol)
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


function combine(basis::RPIBasis, coeffs)

   picoeffs = ntuple(iz0 -> (coeffs[basis.Bz0inds[iz0]]' * basis.A2Bmaps[iz0])[:],
                     numz(basis.pibasis))
   return PIPotential(basis.pibasis, picoeffs)
end


function scaling(basis::RPIBasis, p)
   wwpi = scaling(basis.pibasis, p)
   wwrpi = zeros(Float64, length(basis))
   for iz0 = 1:numz(basis)
      wwpi_iz0 = wwpi[basis.pibasis.inner[iz0].AAindices]
      wwrpi[basis.Bz0inds[iz0]] = basis.A2Bmaps[iz0] * wwpi_iz0
   end
   return wwrpi
end


# ------------------------------------------------------------------------
#    Evaluation code
# ------------------------------------------------------------------------

alloc_temp(basis::RPIBasis, args...) =
   ( AA = site_alloc_B(basis.pibasis, args...),
     AAr = real(site_alloc_B(basis.pibasis, args...)),
     tmp_pibasis = alloc_temp(basis.pibasis, args...)
   )

function evaluate!(B, tmp, basis::RPIBasis, Rs, Zs, z0)
   iz0 = z2i(basis, z0)
   AA = site_evaluate!(tmp.AA, tmp.tmp_pibasis, basis.pibasis, Rs, Zs, z0)
   # TODO: this could be done better maybe by adding the real function into
   #       site_evaluate!, or by writing a real version of it...
   AAr = @view tmp.AAr[1:length(AA)]
   AAr[:] .= real.(AA)
   Bview = @view B[basis.Bz0inds[iz0]]
   mul!(Bview, basis.A2Bmaps[iz0], AAr)
   return B
end

# ------- gradient

alloc_temp_d(basis::RPIBasis, Rs::AbstractVector, args...) =
   alloc_temp_d(basis, length(Rs))

alloc_temp_d(basis::RPIBasis, nmax::Integer) =
    (
    AA = site_alloc_B(basis.pibasis),
    AAr = real(site_alloc_B(basis.pibasis)),
    tmp_pibasis = alloc_temp(basis.pibasis, nmax),
    dAA = site_alloc_dB(basis.pibasis, nmax),
    tmpd_pibasis = alloc_temp_d(basis.pibasis, nmax),
    )

# TODO: evaluate also B??? the interface seems to command it.
function evaluate_d!(B, dB, tmpd, basis::RPIBasis, Rs, Zs, z0)
   iz0 = z2i(basis, z0)
   AA, dAA = tmpd.AA, tmpd.dAA
   site_evaluate_d!(AA, dAA, tmpd.tmpd_pibasis, basis.pibasis, Rs, Zs, z0)
   len = length(basis.pibasis.inner[iz0])
   for i = 1:len
      AA[i] = real(AA[i])
   end
   mul!((@view B[basis.Bz0inds[iz0]]), basis.A2Bmaps[iz0], (@view AA[1:len]))
   for j = 1:length(Rs)
      # ∂∏A / ∂𝐫ⱼ
      dAAj = @view dAA[1:len, j]
      # TODO: check whether copying is faster
      for i = 1:length(dAAj)
         @inbounds dAAj[i] = real.(dAAj[i])
      end
      # copy into B
      dBview = @view dB[basis.Bz0inds[iz0], j]
      mul!(dBview, basis.A2Bmaps[iz0], dAAj)
   end
   return dB
end
