
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

function eval_basis!(B, tmp, ship::SHIPBasis2{T},
                     Rs::AbstractVector{<: JVec},
                     Zs::AbstractVector{<: Integer},
                     z0::Integer ) where {T}
   iz0 = z2i(ship, z0)
   fill!(tmp.Bc, zero(Complex{T}))
   precompute_A!(tmp, ship, Rs, Zs, iz0)
   precompute_AA!(tmp, ship, iz0)
   mul!(tmp.Bc, ship.A2B[iz0], tmp.AA[iz0])
   B .= real.(tmp.Bc)
   return B
end
