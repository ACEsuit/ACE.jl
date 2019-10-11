using StaticArrays, LinearAlgebra
using JuLIP: JVec
import JuLIP
using JuLIP.MLIPs: IPBasis
import JuLIP: alloc_temp, alloc_temp_d

using SHIPs.SphericalHarmonics: SHBasis, sizeY, cart2spher, index_y
using SHIPs.Rotations: ClebschGordan

import Base: Dict, convert, ==

export SHIPBasis


# -------------------------------------------------------------
#       define the basis itself
# -------------------------------------------------------------
# THOUGHTS
#  - technically we don't have to store spec in the basis, but only
#    to generate it?
#  - Possibly remove it entirely  from `SHIPBasis`, but still use it to
#    initially generate that basis.

# TODO [tuples]
# for now ignore 1-body and 2-body, and leave the indexing into
# Nu to mean the number of neighbours. But after this runs, we
# should rewrite this as Nu[1] -> 1-body, Nu[2] -> 2-body, etc.
# so the meaning of BO will return to what it should be.

# TODO: Move to precomputed ∏A coefficients instead of Clebsch-Gordan
#       coefficients to speed up LSQ assembly.

const Tνz{N} = NamedTuple{  (:izz, :ν),
                  Tuple{SVector{N, Int16}, SVector{N, IntS}} }

"""
`struct SHIPBasis` : the main type around eveything in `SHIPs.jl` revolves;
it implements a permutation and rotation invariant basis.

### Developer Docs

* `spec` : degree type specifying which tuples to keep
* `J` : `TransformedJacobi` basis set for the `r`-component
* `SH` : spherical harmonics basis set for the `R̂`-component
* `KL` : list of all admissible `(k,l)` tuples
* `NuZ` : a ν ∈ `Nu[n]` specifies an n-body basis function B_ν = ∑_m ∏_i A_νᵢm
(details see `README.md`)
* `firstA` : same length as `KL`; each `(k,l) = KL[i]` has `2l+1`
A_klm-functions associated which will be stored in the `A` buffer, the first of
these is stored as `A[firstA[i]]`.
* `cg` : precomputed Clebsch Gordan coefficients
"""
struct SHIPBasis{BO, T, NZ, TJ,
                 TSPEC <: BasisSpec{BO, NZ}} <: IPBasis
   spec::TSPEC         # specify which tensor products to keep  in the basis
   J::TJ               # specifies the radial basis
   SH::SHBasis{T}      # specifies the angular basis
   # ------------------------------------------------------------------------
   KL::NTuple{NZ, Vector{NamedTuple{(:k, :l),Tuple{IntS,IntS}}}}    # 1-particle indexing
   NuZ::SMatrix{BO, NZ, Vector}       # N-particle indexing
   firstA::NTuple{NZ, Vector{IntS}}   # indexing into A-basis vectors
   rotcoefs::SVector{BO, Dict}        # storage for the rot-inv coefs
   len_Bll::SMatrix{BO, NZ, Vector{IntS}}  # length of an ll - basis subblock
   idx_Bll::SMatrix{BO, NZ, Vector{IntS}}  # first index of an ll - basis subblock
end

function SHIPBasis(spec::BasisSpec, trans::DistanceTransform, fcut::PolyCutoff;
                   kwargs...)
   J = TransformedJacobi(maxK(spec), trans, fcut)
   return SHIPBasis(spec, J; kwargs...)
end

function SHIPBasis(spec::BasisSpec, trans::DistanceTransformCut; kwargs...)
   J = TransformedJacobi(maxK(spec), trans)
   return SHIPBasis(spec, J; kwargs...)
end


function SHIPBasis(spec::BasisSpec{BO}, J; filter = true, Nsamples = 1_000
                  ) where {BO}
   # precompute the rotation-coefficients
   Bcoefs = Rotations.CoeffArray(Float64)
   # instantiate the basis specification
   allKL, NuZ = generate_ZKL_tuples(spec, Bcoefs)
   # R̂ - basis
   SH = SHBasis(get_maxL(allKL))
   # compute the (l,k) -> indexing into A[(k,l,m)] information
   firstA = _firstA.(allKL)
   # get the Ylm basis coefficients
   rotcoefs = precompute_rotcoefs(allKL, NuZ, Bcoefs)
   len_Bll, idx_Bll = precompute_Bll(allKL, NuZ, rotcoefs)
   # putting it all together ...
   prebasis = SHIPBasis(spec, J, SH, allKL, NuZ, firstA, rotcoefs, len_Bll, idx_Bll)
   if filter
      return filter_dependents(prebasis, Nsamples)
   else
      return prebasis
   end
end

Dict(shipB::SHIPBasis) = Dict(
      "__id__" => "SHIPs_SHIPBasis",
      "spec" => Dict(shipB.spec),
      "J" => Dict(shipB.J) )

SHIPBasis(D::Dict) = SHIPBasis(
      decode_dict(D["spec"]),
      TransformedJacobi(D["J"]) )

convert(::Val{:SHIPs_SHIPBasis}, D::Dict) = SHIPBasis(D)

==(B1::SHIPBasis, B2::SHIPBasis) =
      (B1.spec == B2.spec) && (B1.J == B2.J)

z2i(B::SHIPBasis, z::Integer) = z2i(B.spec, z)
i2z(B::SHIPBasis, i::Integer) = i2z(B.spec, i)

nspecies(B::SHIPBasis) = nspecies(B.spec)

bodyorder(ship::SHIPBasis{BO}) where {BO} = BO + 1

get_maxL(allKL) = maximum( maximum( kl.l for kl in allKL_ )
                           for allKL_ in allKL )


# ----------------------------------------------
#      Computation of the Ylm coefficients
# ----------------------------------------------

_get_ll(KL, νz) = getfield.(KL[νz.ν], :l)

function precompute_rotcoefs(KL, NuZ::SMatrix{BO, NZ},
                             A::Rotations.CoeffArray{T}) where {BO, NZ, T}
   rotcoefs = SVector{BO, Dict}(
                  [ Dict{SVector{N, IntS}, Matrix{T}}() for N = 1:BO ]... )
   for bo = 1:BO, iz = 1:NZ, νz in NuZ[bo, iz]
      ll = _get_ll(KL[iz], νz) # getfield.(KL[iz][νz.ν], :l)
      if !haskey(rotcoefs[bo], ll)
         rotcoefs[bo][ll] = SHIPs.Rotations.basis(A, ll)
      end
   end
   return rotcoefs
end

get_rotcoefs(shipB::SHIPBasis{BO,T}, ll::SVector{N}) where {BO, T, N} =
      (shipB.rotcoefs[N]::Dict{SVector{N, IntS}, Matrix{T}})[ll]


# get_rotcoefs(rotcoeffs::SVector{BO, Dict{<: SVector, Matrix{T}}}, ll::SVector{N}
#             ) where {BO, T, N} =
#       (rotcoefs[N]::Dict{SVector{N, IntS}, Matrix{T}})[ll]
#       # (shipB.rotcoefs[N]::Dict{SVector{N, IntS}, Matrix{T}})[ll]

get_rotcoefs(rotcoeffs::Dict, ll::SVector) = rotcoeffs[ll]

"""
determines the number of rotation-invariant basis functions
at a given ll - tuple. (basically, just reads out the precomputed
coefficients and checks the size of the matrix)
"""
_length_basis(ship::SHIPBasis, ll::StaticVector{N}) where {N} =
      _length_basis(ship.rotcoeffs, ll)

# U is a #mm(ll) x num-basis matrix where each columns defines
# one basis function
_length_basis(rotcoeffs::SVector{BO, <: Dict}, ll::StaticVector{N}) where {BO, N} =
      size(get_rotcoefs(rotcoeffs[N], ll), 2)


"""
for each ll-tuple, we check the length of the ll-sub-basis and
store it in len_Bll.
"""
function precompute_len_Bll(KL, NuZ_N::Vector{Tνz{N}}, rotcoeffs) where {N}
   len_Bll = Vector{IntS}(undef, length(NuZ_N))
   idx_Bll = Vector{IntS}(undef, length(NuZ_N))
   for (i, νz) in enumerate(NuZ_N)
      ll = _get_ll(KL, νz)
      len = _length_basis(rotcoeffs, ll)
      len_Bll[i] = len
   end
   return len_Bll
end

function precompute_Bll(KL, NuZ::SMatrix{BO, NZ}, rotcoeffs) where {BO, NZ}
   len_Bll = SMatrix{BO, NZ}( [ precompute_len_Bll(KL[iz], NuZ[bo, iz], rotcoeffs)
                                for bo = 1:BO, iz = 1:NZ ] )
   # now compute basis indices from the lengths
   idx = 0
   idx_Bll = Matrix{Vector{IntS}}(undef, BO, NZ)
   for bo = 1:BO, iz = 1:NZ
      sumlen = idx .+ cumsum( len_Bll[bo, iz] )
      idx_Bll[bo, iz] = [ IntS[idx]; sumlen[1:end-1] ]
      idx = sumlen[end]
   end
   idx_Bll = SMatrix{BO, NZ}([ idx_Bll[bo, iz]  for bo = 1:BO, iz = 1:NZ ])
   return len_Bll, idx_Bll
end

"""
`_length_basis(ship, bo, iz)` -> length of sub-basis at body-order bo,
species index iz
"""
_length_basis(ship, bo, iz) = sum(ship.len_Bll[bo, iz])

# the length of the basis depends on how many RI-coefficient sets there are
# so we have to be very careful how we define this.
Base.length(ship::SHIPBasis{BO, T, NZ}) where {BO, T, NZ} =
   sum( _length_basis(ship, bo, iz)   for bo = 1:BO, iz = 1:NZ )


# ----------------------------------------------
#      Computation of the A-basis
# ----------------------------------------------

# TODO: rewrite this without reference maxL, maxK!
# TODO: this is written to later allow non-trivial length_A variation
#       across different species
length_A(spec::BasisSpec) =
   [ sum( sizeY(maxL(spec, k)) for k = 0:maxK(spec) )
     for iz = 1:nspecies(spec) ]

alloc_A(spec::BasisSpec) = zeros.(Ref(ComplexF64), length_A(spec))

alloc_dA(spec::BasisSpec) = zeros.(Ref(JVec{ComplexF64}), length_A(spec))

"""
Given a set of KL = { (k,l) } tuples, we would allocate memory for
storing the 1-particle basis values `A[ (k,l,m) ]`.
"""
function _firstA(KL)
   idx = 1
   firstA = zeros(IntS, length(KL) + 1)
   for i = 1:length(KL)
      firstA[i] = idx
      idx += 2 * KL[i].l + 1
   end
   firstA[end] = idx
   return firstA
end

"""
clear out the A-basis storage
"""
function _zero_A!(A::Vector{Vector{T}}) where {T}
   for iz = 1:length(A)
      fill!(A[iz], zero(T))
   end
   return nothing
end


function precompute_A!(tmp,
                       ship::SHIPBasis,
                       Rs::AbstractVector{JVec{T}},
                       Zs::AbstractVector{<: Integer},
                      ) where {T}
   _zero_A!(tmp.A)
   for (iR, (R, Z)) in enumerate(zip(Rs, Zs))
      iz = z2i(ship, Z)
      # evaluate the r-basis and the R̂-basis for the current neighbour at R
      eval_basis!(tmp.J, tmp.tmpJ, ship.J, norm(R))
      eval_basis!(tmp.Y, tmp.tmpY, ship.SH, R)
      # add the contributions to the A_zklm; the indexing into the
      # A array is determined by `ship.firstA` which was precomputed
      for ((k, l), iA) in zip(ship.KL[iz], ship.firstA[iz])
         for m = -l:l
            @inbounds tmp.A[iz][iA+l+m] += tmp.J[k+1] * tmp.Y[index_y(l, m)]
         end
      end
   end
   return nothing
end


function precompute_grads!(tmp,
                           ship::SHIPBasis,
                           Rs::AbstractVector{JVec{T}},
                           Zs::AbstractVector{<:Integer} ) where {T}
   _zero_A!(tmp.A)
   # TODO: re-order these loops => cf. Issue #2
   #        => then can SIMD them and avoid all copying!
   for (iR, (R, Z)) in enumerate(zip(Rs, Zs))
      # ---------- precompute the derivatives of the Jacobi polynomials
      #            and copy into the tmp array
      eval_basis_d!(tmp.J1, tmp.dJ1, tmp.tmpJ, ship.J, norm(R))
      @simd for a = 1:length(tmp.J1)
         @inbounds tmp.J[iR, a] = tmp.J1[a]
         @inbounds tmp.dJ[iR, a] = tmp.dJ1[a]
      end
      # tmp.J[iR,:] .= tmp.J1[:]
      # tmp.dJ[iR,:] .= tmp.dJ1[:]
      # ----------- precompute the Ylm derivatives
      eval_basis_d!(tmp.Y1, tmp.dY1, tmp.tmpY, ship.SH, R)
      @simd for a = 1:length(tmp.Y1)
         @inbounds tmp.Y[iR,a] = tmp.Y1[a]
         @inbounds tmp.dY[iR,a] = tmp.dY1[a]
      end
      # ----------- precompute the A values
      iz = z2i(ship, Z)
      for ((k, l), iA) in zip(ship.KL[iz], ship.firstA[iz])
         for m = -l:l
            @inbounds tmp.A[iz][iA+l+m] += tmp.J1[k+1] * tmp.Y1[index_y(l, m)]
         end
      end
   end
   return tmp
end


# ----------------------------------------------
#      Computation of the B-basis
# ----------------------------------------------

alloc_B(ship::SHIPBasis, args...) = zeros(Float64, length(ship))
alloc_dB(ship::SHIPBasis, N::Integer) = zeros(JVec{Float64}, N, length(ship))
alloc_dB(ship::SHIPBasis, Rs::AbstractVector, args...) = alloc_dB(ship, length(Rs))

alloc_temp(ship::SHIPBasis, args...) = (
      A = alloc_A(ship.spec),
      J = alloc_B(ship.J),
      Y = alloc_B(ship.SH),
      tmpJ = alloc_temp(ship.J),
      tmpY = alloc_temp(ship.SH)
   )

alloc_temp_d(shipB::SHIPBasis, Rs::AbstractVector{<:JVec}, args...) =
      alloc_temp_d(shipB, length(Rs))

function alloc_temp_d(ship::SHIPBasis, N::Integer)
   J1 = alloc_B(ship.J)
   dJ1 = alloc_dB(ship.J)
   Y1 = alloc_B(ship.SH)
   dY1 = alloc_dB(ship.SH)
   return (
         A = alloc_A(ship.spec),
         J = zeros(eltype(J1), N, length(J1)),
        dJ = zeros(eltype(dJ1), N, length(dJ1)),
         Y = zeros(eltype(Y1), N, length(Y1)),
        dY = zeros(eltype(dY1), N, length(dY1)),
        J1 = J1,
       dJ1 = dJ1,
        Y1 = Y1,
       dY1 = dY1,
      tmpJ = alloc_temp_d(ship.J, N),
      tmpY = alloc_temp_d(ship.SH, N)
   )
end


# """
# compute the zeroth index in a B array (basis values) corresponding
# to the N-body subset of the SHIPBasis. This function specifies the
# ordering of the basis set:
# ```
#    iz    N
# ------------
#    1     1
#    1     2
#    1     ...
#    1     BO
#    2     1
#    2     2
#    2     ...
#    ...   ...
# ```
# """
# function _first_B_idx(ship::SHIPBasis{BO}, N, iz0)::Int where {BO}
#    #
#    # TODO : WARNING - this function is now type-unstable
#    #        and probably very slow. We may wish to precompute something here
#    #
#    # compute the first index into the basis
#    idx0 = 0
#    for iz = 1:iz0-1, n = 1:BO
#       idx0 += _length_basis(ship.NuZ[n, iz])
#    end
#    for n = 1:N-1
#       idx0 += _length_basis(ship.NuZ[n, iz0])
#    end
#    return idx0
# end

function _eval_basis!(B, tmp, ship::SHIPBasis{BO, T}, ::Val{N}, iz0,
                      NuZ_N::Vector{Tνz{N}}) where {BO, T, N}
   @assert N <= BO
   # NuZ_N = ship.NuZ[N, iz0]::Vector{Tνz{N}}
   ZKL = ship.KL

   # loop over N-body basis functions
   # A has already been filled in the outer eval_basis!
   for (νz, idx, len) in zip(NuZ_N, ship.idx_Bll[N, iz0], ship.len_Bll[N, iz0])
      ν = νz.ν
      izz = νz.izz
      kk, ll = _kl(ν, izz, ZKL)
      # read the basis coefficients, this is a Vector{T} with the same length
      # as _mrange(ll)
      Ulm = get_rotcoefs(ship, ll)
      @assert len == size(Ulm, 2)
      # b will eventually become B[idx], but we keep it Complex for now
      # so we can do a sanity check that it is in fact real.
      b = zeros(Complex{T}, len)
      for (im, mm) in enumerate(_mrange(ll))    # loops over mᵢ ∈ -lᵢ:lᵢ s.t. ∑mᵢ = 0
         # compute the symmetry prefactor from the CG-coefficients
         um = @view Ulm[im, :] # Complex{T}(clm)
         ∏Alm = one(Complex{T})
         if norm(um) != 0  # TODO: if bm ≈ 0.0; continue; end
            # for (i, (k, l, m, iz)) in enumerate(zip(kk, ll, mm, izz))
            for α = 1:length(kk)
               # TODO: this is the indexing convention used to construct ∏A
               # (feels brittle - maybe rethink it and write a function for it)
               i0 = ship.firstA[izz[α]][ν[α]]
               ∏Alm *= tmp.A[izz[α]][i0 + ll[α] + mm[α]]
            end
            for i = 1:len
               b[i] += um[i] * ∏Alm
            end
         end
      end
      # two little sanity checks which we could run in a debug mode
      # if b == 0.0
      #    @warn("B[idx] == 0!")
      # end
      # if abs(imag(b) / abs(b)) > 1e-10
      #    @warn("b/|b| == $(b/abs(b))")
      # end
      for i = 1:len
         B[idx+i] = real(b[i])
      end
   end
   return nothing
end

_get_NuZ_N(shipB, ::Val{N}, iz) where {N} = shipB.NuZ[N, iz]::Vector{Tνz{N}}

function eval_basis!(B, tmp, ship::SHIPBasis{BO, T},
                     Rs::AbstractVector{<:JVec},
                     Zs::AbstractVector{<: Integer},
                     z0::Integer ) where {BO, T}
   precompute_A!(tmp, ship, Rs, Zs)
   fill!(B, zero(T))
   iz = z2i(ship, z0)
   nfcalls(Val(BO), valN -> _eval_basis!(B, tmp, ship, valN, iz,
                                         _get_NuZ_N(ship, valN, iz)))
   return B
end




function eval_basis_d!(B, dB, tmp, ship::SHIPBasis{BO, T},
                       Rs::AbstractVector{<:JVec{T}},
                       Zs::AbstractVector{<: Integer},
                       z0::Integer ) where {BO, T}
   fill!(B, T(0.0))
   fill!(dB, zero(JVec{T}))
   # all precomputations of "local" gradients
   precompute_grads!(tmp, ship, Rs, Zs)
   iz = z2i(ship, z0)
   nfcalls(Val(BO), valN -> _eval_basis_d!(B, dB, tmp, ship, Rs, Zs, valN, iz,
                                           _get_NuZ_N(ship, valN, iz)))
   return nothing
end

function _eval_basis_d!(B, dB, tmp, ship::SHIPBasis{BO, T}, Rs, Zs,
                         ::Val{N}, iz0,
                         NuZ_N::Vector{Tνz{N}}) where {BO, T, N}
   @assert N <= BO
   # NuZ_N = ship.NuZ[N, iz0]::Vector{Tνz{N}}
   ZKL = ship.KL

   # loop over N-body basis functions
   # for (idx, νz) in enumerate(NuZ_N)
   for (νz, idx, len) in zip(NuZ_N, ship.idx_Bll[N, iz0], ship.len_Bll[N, iz0])
      ν = νz.ν
      izz = νz.izz
      kk, ll = _kl(ν, izz, ZKL)
      Ulm = get_rotcoefs(ship, ll)
      @assert len == size(Ulm, 2)
      for (im, mm) in enumerate(_mrange(ll))    # loops over mᵢ ∈ -lᵢ:lᵢ s.t. ∑mᵢ = 0
         # skip any m-tuples that aren't admissible
         # if abs(mm[end]) > ll[end]; continue; end
         # ------------------------------------------------------------------
         # compute the symmetry prefactor from the CG-coefficients (done above!)
         ulm = @view Ulm[im, :]
         if norm(ulm) != 0
            # ⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯
            # [1] The basis function B_𝐤𝐥 itself
            #     B_𝐤𝐥 = ∑_𝐦 u_{𝐥𝐦} ∏_a A_{kₐlₐmₐ}
            #     the ∑_𝐦 is the `for mm in _mrange` loop
            # TODO: drop this? only compute the gradients?
            ∏Alm = one(Complex{T})
            for β = 1:length(ν)
               i0 = ship.firstA[izz[β]][ν[β]]
               ∏Alm *= tmp.A[izz[β]][i0 + ll[β] + mm[β]] # the k-info is contained in ν[α]
            end
            for i = 1:len
               B[idx+i] += real(∏Alm * ulm[i])
            end
            # ⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯

            # ⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯
            # [2]  The gradients ∂B_{k}{l} / ∂Rⱼ
            #      ∑_a [ ∏_{b ≠ a} A_{kᵦlᵦmᵦ} ] ∂ϕ_{kₐlₐmₐ} / ∂Rⱼ
            for α = 1:length(ν)
               # ∏A_α =  ∏A / A_α   (we could replace this with _dprodA_dAi!)
               ∏A_α = one(Complex{T})
               for β = 1:length(ν)
                  if β != α
                     i0 = ship.firstA[izz[β]][ν[β]]
                     ∏A_α *= tmp.A[izz[β]][i0 + ll[β] + mm[β]]
                  end
               end

               # now compute and write gradients
               ik = kk[α] + 1
               iy = index_y(ll[α], mm[α])
               for j = 1:length(Rs)
                  # ∂ / ∂Rj only contributes if Rj contributed to A[zα]!!
                  if z2i(ship, Zs[j]) == izz[α]
                     R = Rs[j]
                     ∇ϕ_klm = ( tmp.dJ[j, ik] *  tmp.Y[j, iy] * (R/norm(R))
                               + tmp.J[j, ik] * tmp.dY[j, iy] )
                     for i = 1:len
                        dB[j, idx+i] += real(ulm[i] * ∏A_α * ∇ϕ_klm)
                     end
                  end
               end
            end
            # ⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯

         end
         # ------------------------------------------------------------------
      end
   end
   # return B, dB
end




# -------------------------------------------------------------
#       JuLIP Calculators: energies and forces
# -------------------------------------------------------------

# TODO: move all of this into JuLIP.MLIPs

using NeighbourLists: max_neigs, neigs
using JuLIP: Atoms, sites, neighbourlist
using JuLIP.Potentials: neigsz!
import JuLIP: energy, forces, virial, cutoff, site_energy, site_energy_d

cutoff(shipB::SHIPBasis) = cutoff(shipB.J)


function energy(shipB::SHIPBasis, at::Atoms{T}) where {T}
   E = zeros(length(shipB))
   B = alloc_B(shipB)
   nlist = neighbourlist(at, cutoff(shipB))
   maxnR = maxneigs(nlist)
   tmp = alloc_temp(shipB, maxnR)
   tmpRZ = (R = zeros(JVec{T}, maxnR), Z = zeros(Int16, maxnR))
   for i = 1:length(at)
      j, R, Z = neigsz!(tmpRZ, nlist, at, i)
      eval_basis!(B, tmp, shipB, R, Z, at.Z[i])
      E[:] .+= B[:]
   end
   return E
end


function forces(shipB::SHIPBasis, at::Atoms{T}) where {T}
   # precompute the neighbourlist to count the number of neighbours
   nlist = neighbourlist(at, cutoff(shipB))
   maxR = max_neigs(nlist)
   # allocate space accordingly
   F = zeros(JVec{T}, length(at), length(shipB))
   B = alloc_B(shipB)
   dB = alloc_dB(shipB, maxR)
   tmp = alloc_temp_d(shipB, maxR)
   tmpRZ = (R = zeros(JVec{T}, maxR), Z = zeros(Int16, maxR))
   # assemble site gradients and write into F
   for i = 1:length(at)
      j, R, Z = neigsz!(tmpRZ, nlist, at, i)
      eval_basis_d!(B, dB, tmp, shipB, R, Z, at.Z[i])
      for a = 1:length(R)
         F[j[a], :] .-= dB[a, :]
         F[i, :] .+= dB[a, :]
      end
   end
   return [ F[:, iB] for iB = 1:length(shipB) ]
end


function virial(shipB::SHIPBasis, at::Atoms{T}) where {T}
   # precompute the neighbourlist to count the number of neighbours
   nlist = neighbourlist(at, cutoff(shipB))
   maxR = max_neigs(nlist)
   # allocate space accordingly
   V = zeros(JMat{T}, length(shipB))
   B = alloc_B(shipB)
   dB = alloc_dB(shipB, maxR)
   tmp = alloc_temp_d(shipB, maxR)
   tmpRZ = (R = zeros(JVec{T}, maxR), Z = zeros(Int16, maxR))
   # assemble site gradients and write into F
   for i = 1:length(at)
      j, R, Z = neigsz!(tmpRZ, nlist, at, i)
      eval_basis_d!(B, dB, tmp, shipB, R, Z, at.Z[i])
      for iB = 1:length(shipB)
         V[iB] += JuLIP.Potentials.site_virial(dB[:, iB], R)
      end
   end
   return V
end


function _get_neigs(at::Atoms{T}, i0::Integer, rcut) where {T}
   nlist = neighbourlist(at, rcut)
   maxR = maxneigs(nlist)
   tmpRZ = (R = zeros(JVec{T}, maxR), Z = zeros(Int16, maxR))
   j, R, Z = neigsz!(tmpRZ, nlist, at, i0)
   return j, R, Z
end

function site_energy(basis::SHIPBasis, at::Atoms, i0::Integer)
   j, Rs, Zs = _get_neigs(at, i0, cutoff(basis))
   return eval_basis(basis, Rs, Zs, at.Z[i0])
end


function site_energy_d(basis::SHIPBasis, at::Atoms{T}, i0::Integer) where {T}
   Ineigs, Rs, Zs = _get_neigs(at, i0, cutoff(basis))
   dEs = [ zeros(JVec{T}, length(at)) for _ = 1:length(basis) ]
   _, dB = eval_basis_d(basis, Rs, Zs, at.Z[i0])
   @assert dB isa Matrix{JVec{T}}
   @assert size(dB) == (length(Rs), length(basis))
   for iB = 1:length(basis), n = 1:length(Ineigs)
      dEs[iB][Ineigs[n]] += dB[n, iB]
      dEs[iB][i0] -= dB[n, iB]
   end
   return dEs
end



# ------------------------------------------------------------------
#  code to filter out linearly dependent basis functions
#  and at the same time normalise the basis functions
# ------------------------------------------------------------------

"""
main function to call to orthonormalise all basis blocks; the real work
   is carried out in _compute_gramians_N!.
"""
function filter_dependents(shpB::SHIPBasis{BO}, N_samples=1_000) where {BO}
   G = _all_the_gramians(shpB)
   tmp = alloc_temp(shpB)
   B = alloc_B(shpB)
   for N = 1:BO
      _compute_gramians_N!(G, Val(N), shpB, Nsamples, tmp, B)
   end
   newrotcoefs = filter_rotcoefs(shpB, G)
   newlen_Bll, newidx_Bll = precompute_Bll(shpB.allKL, shpB.NuZ, newrotcoefs)
   return SHIPBasis(shpB.spec, shpB.J, shpB.SH, shpB.allKL, shpB.NuZ,
                    shpB.firstA, newrotcoefs, newlen_Bll, newidx_Bll)
end


"""
allocates space for a gramian matrix for every basis block these are then
filled in `_compute_gramians_N!`
"""
function _all_the_gramians(shpB::SHIPBasis{BO, T, NZ}) where {BO, T, NZ}
   G = Matrix{Vector{Matrix{T}}}(undef, BO, NZ)
   for N = 1:BO, iz = 1:NZ
      G[N, iz] = Vector{Matrix{T}}(undef, length(shpB.idx_Bll[N, iz]))
      for j = 1:length(length(shpB.idx_Bll[N, iz]))
         nBll = shpB.len_Bll[N, iz][j]
         G[N, iz][j] = zeros(T, nBll, nBll)
      end
   end
   return G
end



function _compute_gramians_N!(G, ::Val{N}, shpB::SHIPBasis{BO, T, NZ}, Nsamples,
                              tmp, B ) where {N, BO, T, NZ}
   if !NZ == 1
      error("the gramian computation is only implemented for the single-species basis")
   end
   # TODO: for multi-species we need to loop over all combinations of species
   #       this requires a subtle computation of an INZ array?
   #       or maybe we can just compute all of them ... ???

   z0 = shpB.spec.Zs[1]
   Zs = fill(z0, N)
   for ns = 1:Nsamples
      # evaluate all basis functions on a randomly drawn N-body configuration
      Rs = SHIPs.Utils.rand(shpB.J, N)
      eval_basis!(B, tmp, shpB, Rs, Zs, z0)
      # compute the corresponding gramian contributions...
      for iν = 1:length(shpB.idx_Bll)
         B_iν = B[(shpB.idx_Bll[N][iν]+1):(shpB.idx_Bll[N][iν+1])]
         G[N][iν] += B_iν * B_iν' / Nsamples
      end
   end
end

function _get_filtered_basis(shpB, G)

   for
end
