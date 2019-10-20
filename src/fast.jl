
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


using SHIPs.SphericalHarmonics: SHBasis, index_y
using StaticArrays
using JuLIP: AbstractCalculator, Atoms, JVec
using JuLIP.Potentials: MSitePotential, SZList, ZList
using NeighbourLists: max_neigs, neigs

import JuLIP, JuLIP.MLIPs
import JuLIP: cutoff, alloc_temp, alloc_temp_d
import JuLIP.Potentials: evaluate!, evaluate_d!
import JuLIP.MLIPs: combine
import Base: Dict, convert, ==

export SHIP

struct SHIP{T, NZ, TJ} <: MSitePotential
   J::TJ
   SH::SHBasis{T}
   zlist::SZList{NZ}
   # -------------- A and AA datastructures + coefficients
   alists::NTuple{NZ, AList}
   aalists::NTuple{NZ, AAList}
   coeffs::NTuple{NZ, Vector{Complex{T}}}
end


cutoff(ship::SHIP) = cutoff(ship.J)


==(S1::SHIP, S2::SHIP) =
      all( getfield(S1, i) == getfield(S2, i)
           for i = 1:fieldcount(SHIP) )


Base.length(ship::SHIP) = sum(length.(ship.aalists))

# BO + 1 because BO is the number of neighbours not the actual body-order
bodyorder(ship::SHIP{BO}) where {BO} = BO + 1


# ------------------------------------------------------------
#   Initialisation code
# ------------------------------------------------------------

combine(basis::SHIPBasis, coeffs) = SHIP(basis, coeffs)

function SHIP(basis::SHIPBasis{T, NZ}, coeffs::AbstractVector{T}
              ) where {T, NZ}
   AAcoeffs = ntuple(iz0 -> (coeffs[_get_I_iz0(basis, iz0)]' * basis.A2B[iz0])[:], NZ)
   return SHIP( basis.J, basis.SH, basis.zlist,
                basis.alists, basis.aalists, AAcoeffs )
end


# ------------------------------------------------------------
#   FIO code
# ------------------------------------------------------------

Dict(ship::SHIP{T,NZ}) where {T, NZ} = Dict(
      "__id__" => "SHIPs_SHIP_v2",
      "J" => Dict(ship.J),
      "SH_maxL" => ship.SH.maxL,   # TODO: replace this with Dict(SH)
      "T" => string(eltype(ship.SH)),
      "zlist" => Dict(ship.zlist),
      "alists" => [Dict.(ship.alists)...],
      "aalists" => [Dict.(ship.aalists)...],
      "coeffs_re" => [ real.(ship.coeffs[i]) for i = 1:NZ  ],
      "coeffs_im" => [ imag.(ship.coeffs[i]) for i = 1:NZ  ]
   )

convert(::Val{:SHIPs_SHIP_v2}, D::Dict) = SHIP(D)

# bodyorder - 1 is because BO is the number of neighbours
# not the actual body-order
function SHIP(D::Dict)
   T = Meta.eval(Meta.parse(D["T"]))
   J = TransformedJacobi(D["J"])
   SH = SHBasis(D["SH_maxL"], T)
   zlist = decode_dict(D["zlist"])
   NZ = length(zlist)
   alists = ntuple(i -> AList(D["alists"][i]), NZ)
   aalists = ntuple(i -> AAList(D["aalists"][i], alists[i]), NZ)
   coeffs = ntuple(i -> T.(D["coeffs_re"][i]) + im * T.(D["coeffs_im"][i]), NZ)
   return  SHIP(J, SH, zlist, alists, aalists, coeffs)
end


# ------------------------------------------------------------
#   Evaluation code
# ------------------------------------------------------------



alloc_temp(ship::SHIP{T,NZ}, N::Integer) where {T, NZ} =
   (     J = alloc_B(ship.J),
         Y = alloc_B(ship.SH),
         A = [ zeros(Complex{T}, length(ship.alists[iz])) for iz=1:NZ ],
      tmpJ = alloc_temp(ship.J),
      tmpY = alloc_temp(ship.SH),
         R = zeros(JVec{T}, N),
         Z = zeros(Int16, N)
           )



# compute one site energy
function evaluate!(tmp, ship::SHIP{T},
                   Rs::AbstractVector{JVec{T}},
                   Zs::AbstractVector{<:Integer},
                   z0::Integer) where {T}
   iz0 = z2i(ship, z0)
   precompute_A!(tmp.A[iz0], tmp, ship.alists[iz0], Rs, Zs, ship)
   aalist = ship.aalists[iz0]
   A = tmp.A[iz0]
   c = ship.coeffs[iz0]
   Es = zero(T)
   for iAA = 1:length(aalist)
      Esi = c[iAA] # one(Complex{T})
      for α = 1:aalist.len[iAA]
         Esi *= A[aalist.i2Aidx[iAA, α]]
      end
      Es += real(Esi)
   end
   return Es
end


alloc_temp_d(ship::SHIP{T, NZ}, N::Integer) where {T, NZ} =
      ( J = alloc_B(ship.J),
       dJ = alloc_dB(ship.J),
        Y = alloc_B(ship.SH),
       dY = alloc_dB(ship.SH),
        A = [ zeros(Complex{T}, length(ship.alists[iz])) for iz=1:NZ ],
     dAco = [ zeros(Complex{T}, length(ship.alists[iz])) for iz=1:NZ ],
     tmpJ = alloc_temp_d(ship.J),
     tmpY = alloc_temp_d(ship.SH),
       dV = zeros(JVec{T}, N),
        R = zeros(JVec{T}, N),
        Z = zeros(Int16, N)
      )

# compute one site energy
function evaluate_d!(dEs, tmp, ship::SHIP{T},
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
      eval_basis_d!(tmp.J, tmp.dJ, tmp.tmpJ, ship.J, norm(R))
      eval_basis_d!(tmp.Y, tmp.dY, tmp.tmpY, ship.SH, R)
      iz = z2i(ship, Z)
      for iA = alist.firstz[iz]:(alist.firstz[iz+1]-1)
         zklm = alist[iA]
         ik, iy = zklm.k+1, index_y(zklm.l, zklm.m)
         dEs[iR] += real(dAco[iA] * (
                         tmp.J[ik] * tmp.dY[iy] + tmp.dJ[ik] * tmp.Y[iy] * R̂) )
      end
   end
   return dEs
end

function _evaluate_d_stage2!(dAco::AbstractVector{CT}, A, aalist, c, ship
                            ) where {CT}
   fill!(dAco, 0)
   for iAA = 1:length(aalist)
      for α = 1:aalist.len[iAA]
         CxA_α = CT(c[iAA])
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
