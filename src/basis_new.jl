
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------

using StaticArrays, LinearAlgebra
using JuLIP: JVec
import JuLIP
using JuLIP.MLIPs: IPBasis
using JuLIP.Potentials: SZList, ZList
import JuLIP: alloc_temp, alloc_temp_d

using SHIPs.SphericalHarmonics: SHBasis, sizeY, cart2spher, index_y
using SHIPs.Rotations: ClebschGordan
using SparseArrays: SparseMatrixCSC, sparse

import Base: Dict, convert, ==

export SHIPBasis2



struct SHIPBasis2{T, NZ, TJ} <: IPBasis
   J::TJ               # specifies the radial basis
   SH::SHBasis{T}      # specifies the angular basis
   # ------------------------------------------------------------------------
   bgrps::NTuple{NZ, Vector{Tuple}}  # specification of basis functions
   zlist::SZList{NZ}                 # list of species (S=static)
   # ------------------------------------------------------------------------
   alists::NTuple{NZ, AList}
   aalists::NTuple{NZ, AAList}
   A2B::NTuple{NZ, SparseMatrixCSC{Complex{T}, IntS}}
end

function SHIPBasis2(spec::BasisSpec, trans::DistanceTransform, fcut::PolyCutoff;
                   kwargs...)
   J = TransformedJacobi(maxK(spec), trans, fcut)
   return SHIPBasis2(spec, J; kwargs...)
end

function SHIPBasis2(spec::BasisSpec{BO}, J;
                    filter = false, Nsamples = 1_000, pure = false, T = Float64
                   ) where {BO}
   # precompute the rotation-coefficients
   Bcoefs = Rotations.CoeffArray(T)
   # instantiate the basis specification
   allKL, NuZ = generate_ZKL_tuples(spec, Bcoefs)
   # R̂ - basis
   SH = SHBasis(get_maxL(allKL))
   # get the Ylm basis coefficients
   rotcoefs = precompute_rotcoefs(allKL, NuZ, Bcoefs)
   # convert to new format ...
   bgrps = convert_basis_groups(NuZ, allKL) # zkl tuples
   alists, aalists = alists_from_bgrps(bgrps)        # zklm tuples, A, AA
   A2B = A2B_matrices(bgrps, alists, aalists, rotcoefs, T)
   Zs = spec.Zs
   @assert issorted(Zs)
   zlist = ZList([Zs...]; static=true)
   return SHIPBasis2( J, SH,
                      bgrps, zlist,
                      alists, aalists, A2B )
end





function SHIPBasis2(shpB1::SHIPBasis{BO, T}) where {BO, T}
   bgrps = convert_basis_groups(shpB1.NuZ, shpB1.KL) # zkl tuples
   alists, aalists = alists_from_bgrps(bgrps)        # zklm tuples, A, AA
   rotcoefs = shpB1.rotcoefs
   A2B = A2B_matrices(bgrps, alists, aalists, rotcoefs, T)
   Zs = shpB1.spec.Zs
   @assert issorted(Zs)
   zlist = ZList([Zs...]; static=true)
   return SHIPBasis2( shpB1.J, shpB1.SH,
                      bgrps, zlist,
                      alists, aalists, A2B )
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
   # precompute_AA!(tmp, ship, iz0)
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
# linking to the functions implemented in `Alist.jl`


precompute_A!(tmp, ship::SHIPBasis2, Rs, Zs, iz0) =
   precompute_A!(tmp.A[iz0], tmp, ship.alists[iz0], Rs, Zs, ship)

precompute_dA!(tmp, ship::SHIPBasis2,
                    Rs::AbstractVector{<:JVec},
                    Zs::AbstractVector{<:Integer}, iz0 ) =
   precompute_dA!(tmp.A[iz0], tmp.dA[iz0], tmp, ship.alists[iz0],
                  Rs, Zs, ship)

precompute_AA!(tmp, ship::SHIPBasis2, iz0) =
   precompute_AA!(tmp.AA[iz0], tmp.A[iz0], ship.aalists[iz0])
