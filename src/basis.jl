
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
using LinearAlgebra: svd

import Base: Dict, convert, ==

export SHIPBasis, bodyorder



"""
`struct SHIPBasis` : one of the two main types provided by `SHIPs.jl`;
represents a SHIP basis.

The standard constructor is
```
SHIPBasis(spec::BasisSpec, trans::DistanceTransform, fcut::PolyCutoff; kwargs...)
```
Keyword arguments are
* filter = true
* pure = false
* T = Float64
"""
struct SHIPBasis{T, NZ, TJ} <: IPBasis
   J::TJ               # specifies the radial basis
   SH::SHBasis{T}      # specifies the angular basis
   # ------------------------------------------------------------------------
   bgrps::NTuple{NZ, Vector{Tuple}}  # specification of basis functions
                                     # each group def. by a tuple (izz, kk, ll)
   zlist::SZList{NZ}                 # list of species (S=static)
   # ------------------------------------------------------------------------
   alists::NTuple{NZ, AList}         # datastructure to assemble A
   aalists::NTuple{NZ, AAList}       # datastructure to assemble AA
   A2B::NTuple{NZ, SparseMatrixCSC{Complex{T}, IntS}}   # convert AA -> B
   firstb::NTuple{NZ, Vector{IntS}}  # for each bgrp, store the zeroth index of
                                     # in the B vector; firstb[iz0][end] is the
                                     # last index
end

# ---------------(de-)dictionisation---------------------------------
Dict(shipB::SHIPBasis) = Dict(
      "__id__" => "SHIPs_SHIPBasis_v3",
      "J" => Dict(shipB.J),
      "zlist" => Dict(shipB.zlist),
      "bgrps" => bgrp2vecvec.(shipB.bgrps)
   )
convert(::Val{:SHIPs_SHIPBasis_v3}, D::Dict) = SHIPBasis(D)
SHIPBasis(D::Dict) = SHIPBasis(TransformedJacobi(D["J"]),
                               SZList(D["zlist"]),
                               vecvec2bgrps(D["bgrps"]))
bgrp2vecvec(bgrp) = [ Vector{Int}(vcat(b...)) for b in bgrp ]
vecvec2bgrps(Vbs) = convert.(Vector{Tuple}, tuple(vecvec2bgrp.(Vbs)...))
vecvec2bgrp(Vb) = _vec2b.(Vb)
_vec2b(v::Vector{<:Integer}) = (
   N = length(v) ÷ 3; ( SVector{N, Int16}(v[1:N]...),
                        SVector{N,  IntS}(v[N+1:2*N]...),
                        SVector{N,  IntS}(v[2*N+1:3*N]...) ) )

==(B1::SHIPBasis, B2::SHIPBasis) =
      (B1.J == B2.J) && (B1.bgrps == B2.bgrps) && (B1.zlist == B2.zlist)
# ------------------------------------------------------------------------


function SHIPBasis(spec::BasisSpec, trans::DistanceTransform, fcut::PolyCutoff;
                   kwargs...)
   J = TransformedJacobi(maxK(spec), trans, fcut)
   return SHIPBasis(spec, J; kwargs...)
end

function SHIPBasis(spec::BasisSpec, J; T=Float64, kwargs...)
   # precompute the rotation-coefficients
   Bcoefs = Rotations.CoeffArray(T)
   # instantiate the basis specification
   allKL, NuZ = generate_ZKL_tuples(spec, Bcoefs)
   # # get the Ylm basis coefficients
   # rotcoefs = precompute_rotcoefs(allKL, NuZ, Bcoefs)
   # convert to new format ...
   bgrps = convert_basis_groups(NuZ, allKL) # zkl tuples
   # z-list to get i2z and z2i maps
   Zs = spec.Zs
   @assert issorted(Zs)
   zlist = ZList([Zs...]; static=true)

   return SHIPBasis(J, zlist, bgrps; T=T, Bcoefs=Bcoefs, kwargs...)
end

function SHIPBasis(J, zlist::SZList, bgrps::NTuple{NZ, Vector{Tuple}};
                   filter = true, T = Float64, pure = false,
                   Bcoefs = Rotations.CoeffArray(T)
                   ) where {NZ}
   @assert pure == false
   SH = SHBasis(get_maxL(bgrps))
   alists, aalists = alists_from_bgrps(bgrps)        # zklm tuples, A, AA
   A2B, firstb = A2B_matrices(bgrps, alists, aalists, Bcoefs, T)
   preB = SHIPBasis( J, SH,
                     bgrps, zlist,
                     alists, aalists, A2B, firstb )
   if filter
      return alg_filter_rpi_basis(preB)
   end
   return preB
end



function A2B_matrices(bgrps, alists, aalists, Bcoefs, T=Float64)
   NZ = length(alists)
   A2B = Vector{Any}(undef, NZ)
   firstb = Vector{Vector{IntS}}(undef, NZ)
   idx0 = 0
   for iz0  = 1:NZ
      A2B_iz0, firstb_iz0 = A2B_matrix(bgrps[iz0], alists[iz0], aalists[iz0],
                                       Bcoefs, T)
      A2B[iz0] = A2B_iz0
      # firstb_iz0 gives indexing only within the iz0 groups; adding
      # idx0 gives the global basis indices
      firstb[iz0] = idx0 .+ firstb_iz0
      idx0 = firstb[iz0][end]
   end
   return ntuple(i->A2B[i], NZ), ntuple(i->firstb[i], NZ)
end

function A2B_matrix(bgrps, alist, aalist, Bcoefs, T=Float64)
   # allocate triplet format
   Irow, Jcol, vals = IntS[], IntS[], Complex{T}[]
   firstb = IntS[]
   idxB = 0
   # loop through all (zz, kk, ll) tuples; each specifies 1 to several B
   for (izz, kk, ll) in bgrps
      # store the zeroth index for this basis group
      push!(firstb, idxB)
      # get the rotation-coefficients for this basis group
      Ull = SHIPs.Rotations.basis(Bcoefs, ll)
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
   push!(firstb, idxB)
   # create CSC: [   triplet    ]  nrows   ncols
   return sparse(Irow, Jcol, vals, idxB, length(aalist)), firstb
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

# get_maxL(allKL) = maximum( maximum( kl.l for kl in allKL_ )
#                            for allKL_ in allKL )

get_maxL(bgrps::NTuple{NZ, Vector{Tuple}}) where {NZ} = maximum(get_maxL, bgrps)
get_maxL(bgrp::Vector{Tuple}) = maximum(maximum(zkl[3]) for zkl in bgrp)

# the length of the basis depends on how many RI-coefficient sets there are
# so we have to be very careful how we define this.
Base.length(ship::SHIPBasis) = sum(size(A2B, 1) for A2B in ship.A2B)


# ----------------------------------------------
#      Computation of the B-basis
# ----------------------------------------------


alloc_B(ship::SHIPBasis, args...) = zeros(Float64, length(ship))
alloc_dB(ship::SHIPBasis, N::Integer) = zeros(JVec{Float64}, N, length(ship))
alloc_dB(ship::SHIPBasis, Rs::AbstractVector, args...) = alloc_dB(ship, length(Rs))

alloc_temp(ship::SHIPBasis{T, NZ}, args...) where {T, NZ} = (
      A = [ alloc_A(ship.alists[iz0])  for iz0 = 1:NZ ],
      AA = [ alloc_AA(ship.aalists[iz0])  for iz0 = 1:NZ ],
      Bc = [ zeros(Complex{T}, size(ship.A2B[iz0], 1)) for iz0=1:NZ],
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
       dBc = [zeros(JVec{Complex{T}}, size(ship.A2B[iz0], 1)) for iz0=1:NZ],
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

"""
return all basis indices corresponding to species index iz0
"""
function _get_I_iz0(ship::SHIPBasis, iz0)
   idx0 = 0
   for iz = 1:iz0-1
      idx0 += size(ship.A2B[iz], 1)
   end
   idxend = idx0 + size(ship.A2B[iz0], 1)
   I_iz0 = (idx0+1):idxend
   @assert I_iz0 == (ship.firstb[iz0][1]+1):ship.firstb[iz0][end]
   return I_iz0
end

# _get_I_iz0(ship::SHIPBasis, iz0) = (ship.firstb[iz0][1]+1):ship.firstb[iz0][end]



function eval_basis!(B, tmp, ship::SHIPBasis{T},
                     Rs::AbstractVector{<: JVec},
                     Zs::AbstractVector{<: Integer},
                     z0::Integer ) where {T}
   fill!(B, 0)
   iz0 = z2i(ship, z0)
   precompute_A!(tmp, ship, Rs, Zs, iz0)
   precompute_AA!(tmp, ship, iz0)
   _my_mul!(tmp.Bc[iz0], ship.A2B[iz0], tmp.AA[iz0])
   Iz0 = _get_I_iz0(ship, iz0)
   B[Iz0] .= real.(tmp.Bc[iz0])
   return B
end



function eval_basis_d!(dB, tmp, ship::SHIPBasis{T},
                       Rs::AbstractVector{<: JVec},
                       Zs::AbstractVector{<: Integer},
                       z0::Integer ) where {T}
   fill!(dB, zero(JVec{T}))
   iz0 = z2i(ship, z0)
   len_AA = length(ship.aalists[iz0])
   precompute_dA!(tmp, ship, Rs, Zs, iz0)
   for j = 1:length(Rs)
      dAAj = grad_AA_Rj!(tmp, ship, j, Rs, Zs, iz0)  # writes into tmp.dAAj[iz0]
      _my_mul!(tmp.dBc[iz0], ship.A2B[iz0], dAAj)
      Iz0 = _get_I_iz0(ship, iz0)
      @inbounds for i = 1:length(tmp.dBc[iz0])
         dB[j, Iz0[i]] = real(tmp.dBc[iz0][i])
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
#       Filtering the extra basis functions
# -------------------------------------------------------------

len_bgrp(shpB::SHIPBasis, igrp, iz0) =
      shpB.firstb[iz0][igrp+1] - shpB.firstb[iz0][igrp]

alllen_bgrp(shpB::SHIPBasis, iz0) =
      [len_bgrp(shpB, i, iz0) for i = 1:length(shpB.bgrps[iz0])]

I_bgrp(shpB::SHIPBasis, igrp, iz0) =
      (shpB.firstb[iz0][igrp]+1):shpB.firstb[iz0][igrp+1]

function _algebraic_gramian(ship, zkl, Igr, U, iz0)
   izz, kk, ll = zkl
   N = length(kk)
   n = length(Igr)
   @assert size(U, 1) == n
   G = zeros(n, n)
   for σ in permutations(1:N)
      if (izz[σ] != izz) || (kk[σ] != kk) || (ll[σ] != ll); continue; end
      for mm1 in SHIPs._mrange(ll), mm2 in SHIPs._mrange(ll)
         if mm1[σ] == mm2
            iU1 = ship.aalists[iz0][(izz, kk, ll, mm1)]
            iU2 = ship.aalists[iz0][(izz, kk, ll, mm2)]
            for i1 = 1:n, i2 = 1:n
               G[i1, i2] += conj(U[i1, iU1]) * U[i2, iU2]
            end
         end
      end
   end
   return G
end


function _alg_filter_group(ship::SHIPBasis{T}, zkl, Igr, U, iz0) where {T}
   G = _algebraic_gramian(ship, zkl, Igr, U, iz0)
   S = svd(G)
   rk = rank(G)
   UT = convert(SparseMatrixCSC{Complex{T},IntS}, S.U[:, 1:rk]')
   return Diagonal(sqrt.(S.S[1:rk])) * UT * U
end


function alg_filter_rpi_basis(preB::SHIPBasis{T, NZ}) where {T, NZ}

   new_A2B = [ sparse(IntS[], IntS[], Complex{T}[], 0, size(preB.A2B[1], 2))
               for iz0 = 1:NZ ]
   new_firstb = [ IntS[] for iz0 = 1:NZ ]
   bidx0 = 0

   for iz0 = 1:NZ
      for igrp = 1:length(preB.bgrps[iz0])
         zkl = preB.bgrps[iz0][igrp]
         Ib_grp =  I_bgrp(preB, igrp, iz0) .- preB.firstb[iz0][1]
         U = preB.A2B[iz0][Ib_grp, :]
         Ufiltered = _alg_filter_group(preB, zkl, Ib_grp, U, iz0)
         new_A2B[iz0] = vcat(new_A2B[iz0], Ufiltered)
         push!(new_firstb[iz0], bidx0)
         bidx0 += size(Ufiltered, 1)
      end
      # and then add one more to get the total length of the basis
      push!(new_firstb[iz0], bidx0)
      # double-check that we have exactly the right number of firstb entries
      @assert length(new_firstb[iz0]) == length(preB.firstb[1])
   end

   return SHIPBasis(preB.J, preB.SH, preB.bgrps, preB.zlist,
                    preB.alists, preB.aalists,
                    ntuple(i->new_A2B[i], NZ),
                    ntuple(i->new_firstb[i], NZ) )
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
   dB = alloc_dB(basis, length(Rs))
   tmp = alloc_temp_d(basis, length(Rs))
   eval_basis_d!(dB, tmp, basis, Rs, Zs, at.Z[i0])
   @assert dB isa Matrix{JVec{T}}
   @assert size(dB) == (length(Rs), length(basis))
   for iB = 1:length(basis), n = 1:length(Ineigs)
      dEs[iB][Ineigs[n]] += dB[n, iB]
      dEs[iB][i0] -= dB[n, iB]
   end
   return dEs
end
