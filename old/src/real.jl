
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------


using ACE.SphericalHarmonics: RSHBasis, index_y
using StaticArrays
using JuLIP: AbstractCalculator, Atoms, JVec
using JuLIP.Potentials: MSitePotential, SZList, ZList
using NeighbourLists: neigs

import JuLIP, JuLIP.MLIPs
import JuLIP: cutoff, alloc_temp, alloc_temp_d
import JuLIP.Potentials: evaluate!, evaluate_d!
import JuLIP.MLIPs: combine
import Base: Dict, convert, ==

export RSHIP

struct RSHIP{T, NZ, TJ} <: MSitePotential
   J::TJ
   SH::RSHBasis{T}
   zlist::SZList{NZ}
   # -------------- A and AA datastructures + coefficients
   alists::NTuple{NZ, AList}
   aalists::NTuple{NZ, AAList}
   coeffs::NTuple{NZ, Vector{T}}
end


cutoff(ship::RSHIP) = cutoff(ship.J)

==(S1::RSHIP, S2::RSHIP) =
      all( getfield(S1, i) == getfield(S2, i)
           for i = 1:fieldcount(RSHIP) )


Base.length(ship::RSHIP) = sum(length.(ship.aalists))

# BO + 1 because BO is the number of neighbours not the actual body-order
bodyorder(ship::RSHIP{BO}) where {BO} = BO + 1

# ------------------------------------------------------------
#   FIO code
# ------------------------------------------------------------

Dict(ship::RSHIP{T,NZ}) where {T, NZ} = Dict(
      "__id__" => "ACE_RSHIP",
      "J" => Dict(ship.J),
      "SH_maxL" => ship.SH.maxL,   # TODO: replace this with Dict(SH)
      "T" => string(fltype(ship.SH)),
      "zlist" => Dict(ship.zlist),
      "alists" => [Dict.(ship.alists)...],
      "aalists" => [Dict.(ship.aalists)...],
      "coeffs" => [ ship.coeffs[i] for i = 1:NZ  ]
   )

convert(::Val{:ACE_RSHIP}, D::Dict) = RSHIP(D)

# bodyorder - 1 is because BO is the number of neighbours
# not the actual body-order
function RSHIP(D::Dict)
   T = Meta.eval(Meta.parse(D["T"]))
   J = TransformedJacobi(D["J"])
   SH = SHBasis(D["SH_maxL"], T)
   zlist = read_dict(D["zlist"])
   NZ = length(zlist)
   alists = ntuple(i -> AList(D["alists"][i]), NZ)
   aalists = ntuple(i -> AAList(D["aalists"][i], alists[i]), NZ)
   coeffs = ntuple(i -> T.(D["coeffs"][i]), NZ)
   return  RSHIP(J, SH, zlist, alists, aalists, coeffs)
end


# ------------------------------------------------------------
#   Evaluation code
# ------------------------------------------------------------



alloc_temp(ship::RSHIP{T,NZ}, N::Integer) where {T, NZ} =
   (     J = alloc_B(ship.J),
         Y = alloc_B(ship.SH),
         A = [ zeros(T, length(ship.alists[iz])) for iz=1:NZ ],
      tmpJ = alloc_temp(ship.J),
      tmpY = alloc_temp(ship.SH),
         R = zeros(JVec{T}, N),
         Z = zeros(Int16, N),
         AA = [ zeros(T, length(ship.aalists[iz])) for iz=1:NZ ],
           )


# compute one site energy
function evaluate!(tmp, ship::RSHIP{T},
                   Rs::AbstractVector{JVec{T}},
                   Zs::AbstractVector{<:Integer},
                   z0::Integer) where {T}
   iz0 = z2i(ship, z0)
   precompute_A!(tmp.A[iz0], tmp, ship.alists[iz0], Rs, Zs, ship)
   aalist = ship.aalists[iz0]
   A = tmp.A[iz0]
   c = ship.coeffs[iz0]
   Es = zero(T)
   @inbounds for iAA = 1:length(aalist)
      Esi = c[iAA]
      for α = 1:aalist.len[iAA]
         Esi *= A[aalist.i2Aidx[iAA, α]]
      end
      Es += Esi
   end
   return Es
end

# compute one site energy
# experimental new code. Not very fast unfortunately ...
function evaluate_new!(tmp, ship::RSHIP{T},
                   Rs::AbstractVector{JVec{T}},
                   Zs::AbstractVector{<:Integer},
                   z0::Integer) where {T}
   iz0 = z2i(ship, z0)
   precompute_A!(tmp.A[iz0], tmp, ship.alists[iz0], Rs, Zs, ship)
   aalist = ship.aalists[iz0]
   A = tmp.A[iz0]
   c = ship.coeffs[iz0]
   AA = tmp.AA[iz0]
   copy!(AA, c)
   for α = 1:length(aalist)
      for iAA = 1:length(aalist)
         if α <= aalist.len[iAA]
            AA[iAA] *= A[aalist.i2Aidx[iAA, α]]
         end
      end
   end
   return sum(AA)
end


alloc_temp_d(ship::RSHIP{T, NZ}, N::Integer) where {T, NZ} =
      ( J = alloc_B(ship.J),
       dJ = alloc_dB(ship.J),
        Y = alloc_B(ship.SH),
       dY = alloc_dB(ship.SH),
        A = [ zeros(T, length(ship.alists[iz])) for iz=1:NZ ],
     dAco = [ zeros(T, length(ship.alists[iz])) for iz=1:NZ ],
     tmpJ = alloc_temp_d(ship.J),
     tmpY = alloc_temp_d(ship.SH),
       dV = zeros(JVec{T}, N),
        R = zeros(JVec{T}, N),
        Z = zeros(Int16, N)
      )

function evaluate_d!(dEs, tmp, ship::RSHIP{T},
                     Rs::AbstractVector{JVec{T}},
                     Zs::AbstractVector{<:Integer},
                     z0::Integer
                     ) where {T}
   iz0 = z2i(ship, z0)
   alist = ship.alists[iz0]
   aalist = ship.aalists[iz0]

   # stage 1: precompute all the A values
   precompute_A!(tmp.A[iz0], tmp, alist, Rs, Zs, ship)

   # stage 2: compute the coefficients for the ∇A_{klm} = ∇ϕ_{klm}
   A = tmp.A[iz0]
   dAco = tmp.dAco[iz0]
   _evaluate_d_stage2!(dAco, A, aalist, ship.coeffs[iz0], ship)

   # stage 3: get the gradients
   fill!(dEs, zero(JVec{T}))
   for (iR, (R, Z)) in enumerate(zip(Rs, Zs))
      R̂ = R / norm(R)
      evaluate_d!(tmp.J, tmp.dJ, tmp.tmpJ, ship.J, norm(R))
      evaluate_d!(tmp.Y, tmp.dY, tmp.tmpY, ship.SH, R)
      iz = z2i(ship, Z)
      for iA = alist.firstz[iz]:(alist.firstz[iz+1]-1)
         zklm = alist[iA]
         ik, iy = zklm.k+1, index_y(zklm.l, zklm.m)
         dEs[iR] += dAco[iA] * (
                         tmp.J[ik] * tmp.dY[iy] + tmp.dJ[ik] * tmp.Y[iy] * R̂)
      end
   end
   return dEs
end

function _evaluate_d_stage2!(dAco::AbstractVector{CT}, A, aalist, c, ship::RSHIP
                            ) where {CT}
   fill!(dAco, 0)
   for iAA = 1:length(aalist)
      for α = 1:aalist.len[iAA]
         CxA_α = c[iAA]
         for β = 1:aalist.len[iAA]
            if β != α
               iAβ = aalist.i2Aidx[iAA, β]
               CxA_α *= A[iAβ]
            end
         end
         iAα = aalist.i2Aidx[iAA, α]
         dAco[iAα] += CxA_α
      end
   end
end
