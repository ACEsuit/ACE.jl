
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

export SHIPBasis


# TODO:
#  - rewrite the SHIPBasis generation from (J, bgrps, zlist) only
#  - (de-)dictionize `bgrps`
#  - (de-)dictionize `SHIPBasis`
#  - documentation

struct SHIPBasis{T, NZ, TJ} <: IPBasis
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



function SHIPBasis(spec::BasisSpec, trans::DistanceTransform, fcut::PolyCutoff;
                   kwargs...)
   J = TransformedJacobi(maxK(spec), trans, fcut)
   return SHIPBasis(spec, J; kwargs...)
end

function SHIPBasis(spec::BasisSpec{BO}, J;
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
   return SHIPBasis( J, SH,
                      bgrps, zlist,
                      alists, aalists, A2B )
end





function SHIPBasis(shpB1::SHIPBasis{BO, T}) where {BO, T}
   bgrps = convert_basis_groups(shpB1.NuZ, shpB1.KL) # zkl tuples
   alists, aalists = alists_from_bgrps(bgrps)        # zklm tuples, A, AA
   rotcoefs = shpB1.rotcoefs
   A2B = A2B_matrices(bgrps, alists, aalists, rotcoefs, T)
   Zs = shpB1.spec.Zs
   @assert issorted(Zs)
   zlist = ZList([Zs...]; static=true)
   return SHIPBasis( shpB1.J, shpB1.SH,
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


nspecies(B::SHIPBasis{T, NZ}) where {T, NZ} = NZ

bodyorder(ship::SHIPBasis) = maximum(bodyorder, ship.aalists)

get_maxL(allKL) = maximum( maximum( kl.l for kl in allKL_ )
                           for allKL_ in allKL )

# the length of the basis depends on how many RI-coefficient sets there are
# so we have to be very careful how we define this.
Base.length(ship::SHIPBasis) = sum(size(A2B, 1) for A2B in ship.A2B)

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

# ----------------------------------------------
#      Computation of the B-basis
# ----------------------------------------------


alloc_B(ship::SHIPBasis, args...) = zeros(Float64, length(ship))
alloc_dB(ship::SHIPBasis, N::Integer) = zeros(JVec{Float64}, N, length(ship))
alloc_dB(ship::SHIPBasis, Rs::AbstractVector, args...) = alloc_dB(ship, length(Rs))

alloc_temp(ship::SHIPBasis{T, NZ}, args...) where {T, NZ} = (
      A = [ alloc_A(ship.alists[iz0])  for iz0 = 1:NZ ],
      AA = [ alloc_AA(ship.aalists[iz0])  for iz0 = 1:NZ ],
      Bc = zeros(Complex{T}, length(ship)),
      J = alloc_B(ship.J),
      Y = alloc_B(ship.SH),
      tmpJ = alloc_temp(ship.J),
      tmpY = alloc_temp(ship.SH)
   )

alloc_temp_d(shipB::SHIPBasis, Rs::AbstractVector{<:JVec}, args...) =
      alloc_temp_d(shipB, length(Rs))


function alloc_temp_d(ship::SHIPBasis{T, NZ}, N::Integer) where {T, NZ}
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


function eval_basis!(B, tmp, ship::SHIPBasis{T},
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



function eval_basis_d!(dB, tmp, ship::SHIPBasis{T},
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


precompute_A!(tmp, ship::SHIPBasis, Rs, Zs, iz0) =
   precompute_A!(tmp.A[iz0], tmp, ship.alists[iz0], Rs, Zs, ship)

precompute_dA!(tmp, ship::SHIPBasis,
                    Rs::AbstractVector{<:JVec},
                    Zs::AbstractVector{<:Integer}, iz0 ) =
   precompute_dA!(tmp.A[iz0], tmp.dA[iz0], tmp, ship.alists[iz0],
                  Rs, Zs, ship)

precompute_AA!(tmp, ship::SHIPBasis, iz0) =
   precompute_AA!(tmp.AA[iz0], tmp.A[iz0], ship.aalists[iz0])




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
   dB = alloc_dB(shipB, maxR)
   tmp = alloc_temp_d(shipB, maxR)
   tmpRZ = (R = zeros(JVec{T}, maxR), Z = zeros(Int16, maxR))
   # assemble site gradients and write into F
   for i = 1:length(at)
      j, R, Z = neigsz!(tmpRZ, nlist, at, i)
      eval_basis_d!(dB, tmp, shipB, R, Z, at.Z[i])
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
   dB = alloc_dB(shipB, maxR)
   tmp = alloc_temp_d(shipB, maxR)
   tmpRZ = (R = zeros(JVec{T}, maxR), Z = zeros(Int16, maxR))
   # assemble site gradients and write into F
   for i = 1:length(at)
      j, R, Z = neigsz!(tmpRZ, nlist, at, i)
      eval_basis_d!(dB, tmp, shipB, R, Z, at.Z[i])
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
   dB = alloc_dB(shipB, length(Rs))
   tmp = alloc_temp_d(shipB, maxR)
   eval_basis_d!(dB, tmp, basis, Rs, Zs, at.Z[i0])
   @assert dB isa Matrix{JVec{T}}
   @assert size(dB) == (length(Rs), length(basis))
   for iB = 1:length(basis), n = 1:length(Ineigs)
      dEs[iB][Ineigs[n]] += dB[n, iB]
      dEs[iB][i0] -= dB[n, iB]
   end
   return dEs
end
