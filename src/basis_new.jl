
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------

using StaticArrays, LinearAlgebra
using JuLIP: JVec
import JuLIP
using JuLIP.MLIPs: IPBasis
import JuLIP: alloc_temp, alloc_temp_d

using SHIPs.SphericalHarmonics: SHBasis, sizeY, cart2spher, index_y
using SHIPs.Rotations: ClebschGordan
using SparseArrays: SparseMatrixCSC, sparse

import Base: Dict, convert, ==

export SHIPBasis2



struct SHIPBasis2{T, NZ, TJ, TSPEC} <: IPBasis
   spec::TSPEC         # specify which tensor products to keep  in the basis
   J::TJ               # specifies the radial basis
   SH::SHBasis{T}      # specifies the angular basis
   # ------------------------------------------------------------------------
   bgrps::NTuple{NZ, Vector{Tuple}}
   alists::NTuple{NZ, AList}
   aalists::NTuple{NZ, AAList}
   A2B::NTuple{NZ, SparseMatrixCSC{Complex{T}, IntS}}
end

function SHIPBasis2(shpB1::SHIPBasis{BO, T}) where {BO, T}
   bgrps = convert_basis_groups(shpB1.NuZ, shpB1.KL) # zkl tuples
   alists, aalists = alists_from_bgrps(bgrps)        # zklm tuples, A, AA
   rotcoefs = shpB1.rotcoefs
   A2B = A2B_matrices(bgrps, alists, aalists, rotcoefs, T)
   return SHIPBasis2( shpB1.spec, shpB1.J, shpB1.SH,
                      bgrps, alists, aalists, A2B )
end

A2B_matrices(bgrps, alists, aalists, rotcoefs, T=Float64) =
       ntuple( iz0 -> A2B_matrix(bgrps[iz0], alists[iz0], aalists[iz0], rotcoefs, T),
               length(alists) )

function A2B_matrix(bgrps, alist, aalist, rotcoefs, T=Float64)
   # allocate triplet format
   Irow, Jcol, vals = IntS[], IntS[], Complex{T}[]
   idxB = 0
   # loop through all (zz, kk, ll) tuples; each specifies 1 to several B
   for (izz, kk, ll) in bgrps
      # get the rotation-coefficients for this basis group
      Ull = rotcoefs[length(ll)][ll]
      # loop over the columns of Ull -> each specifies a basis function
      for ibasis = 1:size(Ull, 2)
         idxB += 1
         # next we loop over the list of admissible mm to write the
         # CG-coefficients into the A2B matrix
         for (im, mm) in enumerate(_mrange(ll))
            # the (izz, kk, ll, mm) tuple corresponds to an entry in the
            # AA vector (for the species iz0) at index idxA:
            idxA = aalist[(izz, kk, ll, mm)]
            push!(Irow, idxB)
            push!(Jcol, idxA)
            push!(vals, Ull[im, ibasis])
         end
      end
   end
   # create CSC: [   triplet    ]  nrows   ncols
   return sparse(Irow, Jcol, vals, idxB, length(aalist))
end


_zkl(νz, ZKL) = (νz.izz, _kl(νz.ν, νz.izz, ZKL)...)

function convert_basis_groups(NuZ, ZKL)
   BO = size(NuZ, 1)
   NZ = size(NuZ, 2)
   @assert NZ == length(ZKL)
   bgrps = ntuple(iz0 -> Tuple[], NZ)
   for iz0 = 1:NZ, νz in vcat(NuZ[:, iz0]...)
      izz, kk, ll = _zkl(νz, ZKL)
      push!(bgrps[iz0], (izz, kk, ll))
   end
   return bgrps
end

# ----------------------------------------


z2i(B::SHIPBasis2, z::Integer) = z2i(B.spec, z)
i2z(B::SHIPBasis2, i::Integer) = i2z(B.spec, i)

nspecies(B::SHIPBasis2{T, NZ}) where {T, NZ} = NZ

# the length of the basis depends on how many RI-coefficient sets there are
# so we have to be very careful how we define this.
Base.length(ship::SHIPBasis2) = sum(size(A2B, 1) for A2B in ship.A2B)

# ----------------------------------------------
#      Computation of the B-basis
# ----------------------------------------------


alloc_B(ship::SHIPBasis2, args...) = zeros(Float64, length(ship))
alloc_dB(ship::SHIPBasis2, N::Integer) = zeros(JVec{Float64}, N, length(ship))
alloc_dB(ship::SHIPBasis2, Rs::AbstractVector, args...) = alloc_dB(ship, length(Rs))

alloc_temp(ship::SHIPBasis2{T, NZ}, args...) where {T, NZ} = (
      A = [ alloc_A(ship.alists[iz0])  for iz0 = 1:NZ ],
      AA = [ alloc_AA(ship.aalists[iz0])  for iz0 = 1:NZ ],
      Bc = zeros(Complex{T}, length(ship)),
      J = alloc_B(ship.J),
      Y = alloc_B(ship.SH),
      tmpJ = alloc_temp(ship.J),
      tmpY = alloc_temp(ship.SH)
   )

alloc_temp_d(shipB::SHIPBasis2, Rs::AbstractVector{<:JVec}, args...) =
      alloc_temp_d(shipB, length(Rs))


function alloc_temp_d(ship::SHIPBasis2{T, NZ}, N::Integer) where {T, NZ}
   J1 = alloc_B(ship.J)
   dJ1 = alloc_dB(ship.J)
   Y1 = alloc_B(ship.SH)
   dY1 = alloc_dB(ship.SH)
   return (
         A = [ alloc_A(ship.alists[iz0])  for iz0 = 1:NZ ],
        dA = [ zeros(JVec{Complex{T}}, N, length(ship.alists[iz0])) for iz0 = 1:NZ ],
        AA = [ alloc_AA(ship.aalists[iz0])  for iz0 = 1:NZ ],
       dBc = zeros(JVec{Complex{T}}, length(ship)),
      dAAj = [ zeros(JVec{Complex{T}}, length(ship.aalists[iz0])) for iz0 = 1:NZ ],
         JJ = zeros(eltype(J1), N, length(J1)),
        dJJ = zeros(eltype(dJ1), N, length(dJ1)),
         YY = zeros(eltype(Y1), N, length(Y1)),
        dYY = zeros(eltype(dY1), N, length(dY1)),
        J = J1,
       dJ = dJ1,
        Y = Y1,
       dY = dY1,
      tmpJ = alloc_temp_d(ship.J, N),
      tmpY = alloc_temp_d(ship.SH, N)
      )
end


function eval_basis!(B, tmp, ship::SHIPBasis2{T},
                     Rs::AbstractVector{<: JVec},
                     Zs::AbstractVector{<: Integer},
                     z0::Integer ) where {T}
   iz0 = z2i(ship, z0)
   precompute_A!(tmp, ship, Rs, Zs, iz0)
   precompute_AA!(tmp, ship, iz0)
   # fill!(tmp.Bc, 0)
   _my_mul!(tmp.Bc, ship.A2B[iz0], tmp.AA[iz0])
   B .= real.(tmp.Bc)
   return B
end



function eval_basis_d!(dB, tmp, ship::SHIPBasis2{T},
                       Rs::AbstractVector{<: JVec},
                       Zs::AbstractVector{<: Integer},
                       z0::Integer ) where {T}
   iz0 = z2i(ship, z0)
   len_AA = length(ship.aalists[iz0])
   precompute_dA!(tmp, ship, Rs, Zs, iz0)
   precompute_AA!(tmp, ship, iz0)
   for j = 1:length(Rs)
      dAAj = grad_AA_Rj!(tmp, ship, j, Rs, Zs, iz0)  # writes into tmp.dAAj[iz0]
      # fill!(tmp.dBc, zero(JVec{Complex{T}}))
      _my_mul!(tmp.dBc, ship.A2B[iz0], dAAj)
      @inbounds for i = 1:length(tmp.dBc)
         dB[j, i] = real(tmp.dBc[i])
      end
   end
   return dB
end



# -------------------------------------------------------
# move back into Alist???


"""
This fills the A-array stored in tmp with the A_zklm density projections in
the order specified by AList. It also evaluates the radial and angular basis
functions along the way.
"""
function precompute_A!(tmp, ship::SHIPBasis2{T}, Rs, Zs, iz0) where {T}
   alist = ship.alists[iz0]
   fill!(tmp.A[iz0], zero(Complex{T}))
   for (R, Z) in zip(Rs, Zs)
      # evaluate the r-basis and the R̂-basis for the current neighbour at R
      eval_basis!(tmp.J, tmp.tmpJ, ship.J, norm(R))
      eval_basis!(tmp.Y, tmp.tmpY, ship.SH, R)
      # add the contributions to the A_zklm
      iz = z2i(ship, Z)
      for i = alist.firstz[iz]:(alist.firstz[iz+1]-1)
         zklm = alist[i]
         tmp.A[iz0][i] += tmp.J[zklm.k+1] * tmp.Y[index_y(zklm.l, zklm.m)]
      end
   end
   return nothing
end


function precompute_dA!(tmp,
                        ship::SHIPBasis2{T},
                        Rs::AbstractVector{JVec{T}},
                        Zs::AbstractVector{<:Integer}, iz0 ) where {T}
   alist = ship.alists[iz0]
   fill!(tmp.A[iz0], zero(Complex{T}))

   for (iR, (R, Z)) in enumerate(zip(Rs, Zs))
      # ---------- precompute the derivatives of the Jacobi polynomials
      #            and copy into the tmp array
      eval_basis_d!(tmp.J, tmp.dJ, tmp.tmpJ, ship.J, norm(R))
      @simd for a = 1:length(tmp.J)
         @inbounds tmp.JJ[iR, a] = tmp.J[a]
         @inbounds tmp.dJJ[iR, a] = tmp.dJ[a]
      end
      # ----------- precompute the Ylm derivatives
      eval_basis_d!(tmp.Y, tmp.dY, tmp.tmpY, ship.SH, R)
      @simd for a = 1:length(tmp.Y)
         @inbounds  tmp.YY[iR,a] = tmp.Y[a]
         @inbounds tmp.dYY[iR,a] = tmp.dY[a]
      end
      # ----------- precompute the A values
      iz = z2i(ship, Z)
      for i = alist.firstz[iz]:(alist.firstz[iz+1]-1)
         zklm = alist[i]
         tmp.A[iz0][i] += tmp.J[zklm.k+1] * tmp.Y[index_y(zklm.l, zklm.m)]
         # and into dA
         ∇ϕ_zklm = grad_phi_Rj(R, iR, zklm, tmp)
         tmp.dA[iz0][iR, i] = ∇ϕ_zklm
      end
   end

   return tmp
end


function precompute_AA!(tmp, ship::SHIPBasis2{T}, iz0) where {T}
   aalist = ship.aalists[iz0]
   A = tmp.A[iz0]
   AA = tmp.AA[iz0]
   fill!(AA, one(Complex{T}))
   for i = 1:length(aalist)
      for α = 1:aalist.len[i]
         iA = aalist.i2Aidx[i, α]
         AA[i] *= A[iA]
      end
   end
   return nothing
end


# --------------------------------------------------------

function grad_phi_Rj(Rj, j, zklm, tmp)
   ik = zklm.k + 1
   iy = index_y(zklm.l, zklm.m)
   return ( tmp.dJJ[j, ik] *  tmp.YY[j, iy] * (Rj/norm(Rj))
           + tmp.JJ[j, ik] * tmp.dYY[j, iy] )
end

function grad_AA_Ab(iAA, b, alist, aalist, A)
   g = one(eltype(A))
   for a = 1:aalist.len[iAA]
      if a != b
         iA = aalist.i2Aidx[iAA, a]
         g *= A[iA]
      end
   end
   return g
end

function grad_AAi_Rj(iAA, j, Rj::JVec{T}, izj::Integer,
                     alist, aalist, A, AA, dA, tmp) where {T}
   g = zero(JVec{Complex{T}})
   for b = 1:aalist.len[iAA] # body-order
      # A_{n_b} = A[iA]
      iA = aalist.i2Aidx[iAA, b]
      # daa_dab = ∂(∏A_{n_a}) / ∂A_{n_b}
      daa_dab = grad_AA_Ab(iAA, b, alist, aalist, A)
      zklm = alist[iA] # (zklm corresponding to A_{n_b})
      ∇ϕ_zklm = dA[j, iA] # grad_phi_Rj(Rj, j, zklm, tmp)
      g += daa_dab * ∇ϕ_zklm
   end
   return g
end

function grad_AA_Rj!(tmp, ship, j, Rs, Zs, iz0) where {T}
   for iAA = 1:length(ship.aalists[iz0])
      # g = ∂(∏_a A_a) / ∂Rj     # TODO: species needs to go in here as well!
      tmp.dAAj[iz0][iAA] = grad_AAi_Rj(iAA, j, Rs[j], z2i(ship, Zs[j]),
                                       ship.alists[iz0], ship.aalists[iz0],
                                       tmp.A[iz0], tmp.AA[iz0], tmp.dA[iz0],
                                       tmp)
   end
   return tmp.dAAj[iz0]
end
