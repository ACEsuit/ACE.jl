
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

using SHIPs.SphericalHarmonics: SHBasis, sizeY, cart2spher, index_y,
         ClebschGordan

import Base: Dict, convert, ==

export SHIPBasis

# in here we specify body-order specific code so that it doesn't pollute
# the main codebase
#  * filter_tuples
#  * _Bcoeff
include("bodyorders.jl")


_mrange(ll::SVector{BO}) where {BO} =
   CartesianIndices(ntuple( i -> -ll[i]:ll[i], (BO-1) ))

"""
return kk, ll, mrange
where kk, ll is BO-tuples of k and l indices, while mrange is a
cartesian range over which to iterate to construct the basis functions

(note: this is tested for correcteness and speed)
"""
function _klm(ν::StaticVector{BO, T}, KL) where {BO, T}
   kk = SVector( ntuple(i -> KL[ν[i]].k, BO) )
   ll = SVector( ntuple(i -> KL[ν[i]].l, BO) )
   mrange = CartesianIndices(ntuple( i -> -ll[i]:ll[i], (BO-1) ))
   return kk, ll, mrange
end


# TODO [tuples] generate basis functions UP TO A BODY-ORDER
#      we need to have 0 stand for a 1 (i.e. body-order)
#      => decide whether to drop T0.

"""
create a vector of Nu arrays with the right type information
for each body-order
"""
function _generate_Nu(bo::Integer, T=IntS)
   Nu = []
   for n = 1:bo
      push!(Nu, SVector{n, T}[])
   end
   # convert into an SVector to make the length a type parameters
   return SVector(Nu...)
end

function generate_KL_tuples(Deg::AbstractDegree, maxbo::Integer, cg; filter=true)
   # all possible (k, l) pairs
   allKL, degs = generate_KL(Deg)
   # sepatare arrays for all body-orders
   Nu = _generate_Nu(maxbo)
   for N = 1:maxbo
      _generate_KL_tuples!(Nu[N], Deg, cg, allKL, degs; filter=filter)
   end
   return allKL, Nu
end

function _generate_KL_tuples!(Nu::Vector{<: SVector{BO}}, Deg::AbstractDegree,
                             cg, allKL, degs; filter=true) where {BO}
   # the first iterm is just (0, ..., 0)
   # we can choose (k1, l1), (k2, l2) ... by indexing into allKL
   # then we start incrementing until we hit the maximum degree
   # while retaining the ordering ν₁ ≤ ν₂ ≤ …
   lastidx = 0
   ν = @MVector ones(IntS, BO)   # (ones(IntS, bo)...)
   while true
      # check whether the current ν tuple is admissible
      # the first condition is that its max index is small enough
      isadmissible = maximum(ν) <= length(allKL)
      if isadmissible
         # the second condition is that the multivariate degree it defines
         # is small enough => for that we first have to compute the corresponding
         # k and l vectors
         kk, ll, _ = _klm(ν, allKL)
         isadmissible = admissible(Deg, kk, ll)
      end

      # we want to increment `curindex`, but if we've reach the maximum degree
      # then we need to move to the next index down

      # if the current tuple ν has admissible degree ...
      if isadmissible
         # ... then we add it to the stack  ...
         #     (at least if it is an admissible basis function respecting
         #      all the symmetries - this is checked by filter_tuples)
         if !filter || filter_tuples(allKL, ν, cg)
            push!(Nu, SVector(ν))
         end
         # ... and increment it
         lastidx = BO
         ν[lastidx] += 1
      else
         # we have overshot, _deg(ν) > deg; we must go back down, by
         # decreasing the index at which we increment
         if lastidx == 1
            break
         end
         ν[lastidx-1:end] .= ν[lastidx-1] + 1
         lastidx -= 1
      end
   end
   return allKL, Nu
end


# -------------------------------------------------------------
#       define the basis itself
# -------------------------------------------------------------
# THOUGHTS
#  - technically we don't have to store Deg in the basis, but only
#    to generate it?

# TODO [tuples]
# for now ignore 1-body and 2-body, and leave the indexing into
# Nu to mean the number of neighbours. But after this runs, we
# should rewrite this as Nu[1] -> 1-body, Nu[2] -> 2-body, etc.
# so the meaning of BO will return to what it should be.

"""
`struct SHIPBasis` : the main type around eveything in `SHIPs.jl` revolves;
it implements a permutation and rotation invariant basis.

### Developer Docs

* `Deg` : degree type specifying which tuples to keep
* `J` : `TransformedJacobi` basis set for the `r`-component
* `SH` : spherical harmonics basis set for the `R̂`-component
* `KL` : list of all admissible `(k,l)` tuples
* `Nu` : a ν ∈ `Nu[n]` specifies an n-body basis function B_ν = ∑_m ∏_i A_νᵢm
(details see `README.md`)
* `firstA` : same length as `KL`; each `(k,l) = KL[i]` has `2l+1`
A_klm-functions associated which will be stored in the `A` buffer, the first of
these is stored as `A[firstA[i]]`.
* `cg` : precomputed Clebsch Gordan coefficients
"""
struct SHIPBasis{BO, T, TJ, TDEG} <: IPBasis
   Deg::TDEG
   J::TJ
   SH::SHBasis{T}
   KL::Vector{NamedTuple{(:k, :l, :deg),Tuple{IntS,IntS,T}}}
   Nu::SVector{BO, Vector{T1} where T1}
   cg::ClebschGordan{T}
   firstA::Vector{IntS}   # indexing into A
end

# TODO: Move to precomputed ∏A coefficients instead of Clebsch-Gordan
#       coefficients to speed up LSQ assembly.


Dict(shipB::SHIPBasis) = Dict(
      "__id__" => "SHIPs_SHIPBasis",
      "Deg" => Dict(shipB.Deg),
      "bodyorder" => bodyorder(shipB),
      "J" => Dict(shipB.J) )

SHIPBasis(D::Dict) = SHIPBasis(
      decode_dict(D["Deg"]),
      D["bodyorder"] - 1,
      TransformedJacobi(D["J"]) )

convert(::Val{:SHIPs_SHIPBasis}, D::Dict) = SHIPBasis(D)

==(B1::SHIPBasis, B2::SHIPBasis) = (
      (B1.Deg == B2.Deg) &&
      (bodyorder(B1) == bodyorder(B2)) &&
      (B1.J == B2.J) )


length_A(Deg::AbstractDegree) = sum( sizeY(maxL(Deg, k)) for k = 0:maxK(Deg) )

alloc_A(Deg::AbstractDegree) = zeros(ComplexF64, length_A(Deg))
alloc_dA(Deg::AbstractDegree) = zeros(JVec{ComplexF64}, length_A(Deg))

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

function SHIPBasis(Deg::AbstractDegree, bo::Integer,
                   trans::DistanceTransform, fcut::PolyCutoff;
                   filter=true)
   J = TransformedJacobi(maxK(Deg), trans, fcut)
   return SHIPBasis(Deg, bo, J; filter=filter)
end

function SHIPBasis(Deg::AbstractDegree, bo::Integer, J::TransformedJacobi;
                   filter=true)
   # R̂ - basis
   SH = SHBasis(maxL(Deg))
   # precompute the Clebsch-Gordan coefficients
   cg = ClebschGordan(maxL(Deg))
   # get the basis specification
   allKL, Nu = generate_KL_tuples(Deg, bo, cg; filter=filter)
   # compute the (l,k) -> indexing into A information
   firstA = _firstA(allKL)
   # putting it all together ...
   return SHIPBasis(Deg, J, SH, allKL, Nu, cg, firstA)
end


bodyorder(ship::SHIPBasis{BO}) where {BO} = BO + 1

Base.length(ship::SHIPBasis) = length_B(ship)
length_B(ship::SHIPBasis{BO}) where {BO} = sum(length.(ship.Nu))

alloc_B(ship::SHIPBasis, args...) = zeros(Float64, length_B(ship))
alloc_dB(ship::SHIPBasis, N::Integer) = zeros(JVec{Float64}, N, length_B(ship))
alloc_dB(ship::SHIPBasis, Rs::AbstractVector) = alloc_dB(ship, length(Rs))

alloc_temp(ship::SHIPBasis) = (
      A = alloc_A(ship.Deg),
      J = alloc_B(ship.J),
      Y = alloc_B(ship.SH),
      tmpJ = alloc_temp(ship.J),
      tmpY = alloc_temp(ship.SH)
   )

alloc_temp_d(shipB::SHIPBasis, Rs::AbstractVector{<:JVec}) =
      alloc_temp_d(shipB, length(Rs))

function alloc_temp_d(ship::SHIPBasis, N::Integer)
   J1 = alloc_B(ship.J)
   dJ1 = alloc_dB(ship.J)
   Y1 = alloc_B(ship.SH)
   dY1 = alloc_dB(ship.SH)
   return (
         A = alloc_A(ship.Deg),
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

# -------------------------------------------------------------
#       precompute the A arrays
# -------------------------------------------------------------

function precompute_A!(tmp, ship::SHIPBasis, Rs::AbstractVector{JVec{T}}) where {T}
   fill!(tmp.A, T(0.0))
   for (iR, R) in enumerate(Rs)
      # evaluate the r-basis and the R̂-basis for the current neighbour at R
      eval_basis!(tmp.J, tmp.tmpJ, ship.J, norm(R))
      eval_basis!(tmp.Y, tmp.tmpY, ship.SH, R)
      # add the contributions to the A_klm; the indexing into the
      # A array is determined by `ship.firstA` which was precomputed
      for ((k, l), iA) in zip(ship.KL, ship.firstA)
         for m = -l:l
            @inbounds tmp.A[iA+l+m] += tmp.J[k+1] * tmp.Y[index_y(l, m)]
         end
      end
   end
   return nothing
end


function precompute_grads!(tmp, ship::SHIPBasis, Rs::AbstractVector{JVec{T}}) where {T}
   fill!(tmp.A, T(0.0))
   # TODO: re-order these loops => cf. Issue #2
   #        => then can SIMD them and avoid all copying!
   for (iR, R) in enumerate(Rs)
      # ---------- precompute the derivatives of the Jacobi polynomials
      #            and copy into the tmp array
      eval_basis_d!(tmp.J1, tmp.dJ1, tmp.tmpJ, ship.J, norm(R))
      tmp.J[iR,:] .= tmp.J1[:]
      tmp.dJ[iR,:] .= tmp.dJ1[:]
      # ----------- precompute the Ylm derivatives
      eval_basis_d!(tmp.Y1, tmp.dY1, tmp.tmpY, ship.SH, R)
      tmp.Y[iR,:] .= tmp.Y1[:]
      tmp.dY[iR,:] .= tmp.dY1[:]
      # ----------- precompute the A values
      for ((k, l), iA) in zip(ship.KL, ship.firstA)
         for m = -l:l
            @inbounds tmp.A[iA+l+m] += tmp.J[iR, k+1] * tmp.Y[iR, index_y(l, m)]
         end
      end
   end
   return tmp
end


# -------------------------------------------------------------
#       Evaluate the actual basis functions
# -------------------------------------------------------------

"""
compute the zeroth index in a B array (basis values) corresponding
to the N-body subset of the SHIPBasis
"""
function _first_B_idx(ship, N)
   # compute the first index into the basis
   idx0 = 0
   for n = 1:N-1
      idx0 += length(ship.Nu[n])
   end
   return idx0
end

_mvec(::CartesianIndex{0}) = SVector(IntS(0))
_mvec(mpre::CartesianIndex{N}) where {N} =
      SVector(Tuple(mpre)..., - sum(Tuple(mpre)))

function _eval_basis!(B, tmp, ship::SHIPBasis{BO, T}, ::Val{N}) where {BO, T, N}
   @assert N <= BO
   Nu_N = ship.Nu[N]::Vector{SVector{N, IntS}}
   KL = ship.KL
   idx0 = _first_B_idx(ship, N)
   # loop over N-body basis functions
   # A has already been filled in the outer eval_basis!
   for (idx, ν) in enumerate(Nu_N)
      kk, ll, mrange = _klm(ν, KL)
      # b will eventually become B[idx], but we keep it Complex for now
      # so we can do a sanity check that it is in fact real.
      b = zero(ComplexF64)
      @assert mrange == _mrange(ll)
      for mpre in mrange    # this is a cartesian loop over BO-1 indices
         mm = _mvec(mpre)
         # skip any m-tuples that aren't admissible
         if abs(mm[end]) > ll[end]; continue; end
         # compute the symmetry prefactor from the CG-coefficients
         bm = ComplexF64(_Bcoeff(ll, mm, ship.cg))
         if bm != 0
            for (i, (k, l, m)) in enumerate(zip(kk, ll, mm))
               # this is the indexing convention used to construct A
               #  (feels brittle - maybe rethink it and write a function for it)
               i0 = ship.firstA[ν[i]]
                bm *= tmp.A[i0 + l + m]
            end
            b += bm
         end
      end
      # two little sanity checks which we could run in a debug mode
      # if b == 0.0
      #    @warn("B[idx] == 0!")
      # end
      # if abs(imag(b) / abs(b)) > 1e-10
      #    @warn("b/|b| == $(b/abs(b))")
      # end
      B[idx0+idx] = real(b)
   end
   return nothing
end

function eval_basis!(B, tmp, ship::SHIPBasis{BO}, Rs::AbstractVector{<:JVec}
                     ) where {BO}
   precompute_A!(tmp, ship, Rs)
   nfcalls(Val(BO), valN -> _eval_basis!(B, tmp, ship, valN))
   return B
end




function eval_basis_d!(B, dB, tmp, ship::SHIPBasis{BO},
                       Rs::AbstractVector{JVec{T}}) where {BO, T}
   fill!(B, T(0.0))
   fill!(dB, zero(JVec{T}))
   # all precomputations of "local" gradients
   precompute_grads!(tmp, ship, Rs)
   nfcalls(Val(BO), valN -> _eval_basis_d!(B, dB, tmp, ship, Rs, valN))
   return nothing
end

function _eval_basis_d!(B, dB, tmp, ship::SHIPBasis{BO, T}, Rs,
                         ::Val{N}) where {BO, T, N}
   @assert N <= BO
   Nu_N = ship.Nu[N]::Vector{SVector{N, IntS}}
   KL = ship.KL
   idx0 = _first_B_idx(ship, N)
   # loop over N-body basis functions
   for (idx, ν) in enumerate(Nu_N)
      idxB = idx0+idx
      kk, ll, mrange = _klm(ν, KL)
      for mpre in mrange    # this is a cartesian loop over BO-1 indices
         mm = _mvec(mpre)
         # skip any m-tuples that aren't admissible
         if abs(mm[end]) > ll[end]; continue; end
         # ------------------------------------------------------------------
         # compute the symmetry prefactor from the CG-coefficients
         C = _Bcoeff(ll, mm, ship.cg)
         if C != 0
            # ⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯
            # [1] The basis function B_𝐤𝐥 itself
            #     B_𝐤𝐥 = ∑_𝐦 C_{𝐤𝐥𝐦} ∏_a A_{kₐlₐmₐ}
            #     the ∑_𝐦 is the `for mpre in mrange` loop
            CxA = ComplexF64(C)
            for α = 1:length(ν)
               i0 = ship.firstA[ν[α]]
               CxA *= tmp.A[i0 + ll[α] + mm[α]] # the k-info is contained in ν[α]
            end
            B[idxB] += real(CxA)
            # ⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯

            # ⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯
            # [2]  The gradients ∂B_{k}{l} / ∂Rⱼ
            #      ∑_a [ ∏_{b ≠ a} A_{kᵦlᵦmᵦ} ] ∂ϕ_{kₐlₐmₐ} / ∂Rⱼ
            for α = 1:length(ν)
               # CxA_α =  CxA / A_α   (we could replace this with _dprodA_dAi!)
               CxA_α = ComplexF64(C)
               for β = 1:length(ν)
                  if β != α
                     i0 = ship.firstA[ν[β]]
                     CxA_α *= tmp.A[i0 + ll[β] + mm[β]]
                  end
               end

               # now compute and write gradients
               ik = kk[α] + 1
               iy = index_y(ll[α], mm[α])
               for j = 1:length(Rs)
                  R = Rs[j]
                  ∇ϕ_klm = (tmp.dJ[j, ik] *  tmp.Y[j, iy] * (R/norm(R))
                           + tmp.J[j, ik] * tmp.dY[j, iy] )
                  dB[j, idxB] += real(CxA_α * ∇ϕ_klm)
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
import JuLIP: energy, forces, virial, cutoff, site_energy, site_energy_d

cutoff(shipB::SHIPBasis) = cutoff(shipB.J)


function energy(shipB::SHIPBasis, at::Atoms)
   E = zeros(length(shipB))
   B = alloc_B(shipB)
   tmp = alloc_temp(shipB)
   for (i, j, R) in sites(at, cutoff(shipB))
      eval_basis!(B, tmp, shipB, R)
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
   # assemble site gradients and write into F
   for (i, j, R) in sites(nlist)
      eval_basis_d!(B, dB, tmp, shipB, R)
      for a = 1:length(R)
         F[j[a], :] .-= dB[a, :]
         F[i, :] .+= dB[a, :]
      end
   end
   return [ F[:, iB] for iB = 1:length(shipB) ]
end


function virial(shipB::SHIPBasis, at::Atoms)
   # precompute the neighbourlist to count the number of neighbours
   nlist = neighbourlist(at, cutoff(shipB))
   maxR = max_neigs(nlist)
   # allocate space accordingly
   V = zeros(JMatF, length(shipB))
   B = alloc_B(shipB)
   dB = alloc_dB(shipB, maxR)
   tmp = alloc_temp_d(shipB, maxR)
   # assemble site gradients and write into F
   for (i, j, R) in sites(nlist)
      eval_basis_d!(B, dB, tmp, shipB, R)
      for iB = 1:length(shipB)
         V[iB] += JuLIP.Potentials.site_virial(dB[:, iB], R)
      end
   end
   return V
end


function _get_neigs(at::Atoms, i0::Integer, rcut)
   nlist = neighbourlist(at, rcut)
   j, R = neigs(nlist, i0)
   return R, j
end

function site_energy(basis::SHIPBasis, at::Atoms, i0::Integer)
   Rs, _ = _get_neigs(at, i0, cutoff(basis))
   return eval_basis(basis, Rs)
end


function site_energy_d(basis::SHIPBasis, at::Atoms{T}, i0::Integer) where {T}
   Rs, Ineigs = _get_neigs(at, i0, cutoff(basis))
   dEs = [ zeros(JVec{T}, length(at)) for _ = 1:length(basis) ]
   _, dB = eval_basis_d(basis, Rs)
   @assert dB isa Matrix{JVec{T}}
   @assert size(dB) == (length(Rs), length(basis))
   for iB = 1:length(basis), n = 1:length(Ineigs)
      dEs[iB][Ineigs[n]] += dB[n, iB]
      dEs[iB][i0] -= dB[n, iB]
   end
   return dEs
end
