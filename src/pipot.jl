
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


using SHIPs.SphericalHarmonics: SHBasis, index_y
using StaticArrays
using JuLIP: AbstractCalculator, Atoms, JVec
using JuLIP.Potentials: SitePotential, SZList, ZList
using NeighbourLists: neigs

import JuLIP, JuLIP.MLIPs

export PIPotential


"""
`struct PIPotential` : specifies a PIPotential, which is basically defined
through a PIBasis and its coefficients
"""
struct PIPotential{T, NZ, TIN} <: SitePotential
   pibasis::TIN
   coeffs::NTuple{NZ, Vector{T}}
end

cutoff(V::PIPotential) = cutoff(V.pibasis)

==(V1::PIPotential, V2::PIPotential) =
      (V1.pibasis == V2.pibasis) && (V1.coeffs == V2.coeffs)

# TODO: this doesn't feel right ... should be real(T)?
Base.eltype(::PIPotential{T}) where {T} = T

z2i(V::PIPotential, z::AtomicNumber) = z2i(V.pibasis, z)

# ------------------------------------------------------------
#   Initialisation code
# ------------------------------------------------------------

combine(basis::PIBasis, coeffs) = PIPotential(basis, coeffs)


function PIPotential(basis::PIBasis, coeffs::Vector{<: Number})
   coeffs_t = ntuple(iz0 -> coeffs[basis.inner[iz0].AAindices], numz(basis))
   return PIPotential(basis, coeffs_t)
end


# ------------------------------------------------------------
#   FIO code
# ------------------------------------------------------------

# Dict(ship::SHIP{T,NZ}) where {T, NZ} = Dict(
#       "__id__" => "SHIPs_SHIP_v2",
#       "J" => Dict(ship.J),
#       "SH_maxL" => ship.SH.maxL,   # TODO: replace this with Dict(SH)
#       "T" => string(eltype(ship.SH)),
#       "zlist" => Dict(ship.zlist),
#       "alists" => [Dict.(ship.alists)...],
#       "aalists" => [Dict.(ship.aalists)...],
#       "coeffs_re" => [ real.(ship.coeffs[i]) for i = 1:NZ  ],
#       "coeffs_im" => [ imag.(ship.coeffs[i]) for i = 1:NZ  ]
#    )
#
# convert(::Val{:SHIPs_SHIP_v2}, D::Dict) = SHIP(D)
#
# # bodyorder - 1 is because BO is the number of neighbours
# # not the actual body-order
# function SHIP(D::Dict)
#    T = Meta.eval(Meta.parse(D["T"]))
#    J = TransformedJacobi(D["J"])
#    SH = SHBasis(D["SH_maxL"], T)
#    zlist = decode_dict(D["zlist"])
#    NZ = length(zlist)
#    alists = ntuple(i -> AList(D["alists"][i]), NZ)
#    aalists = ntuple(i -> AAList(D["aalists"][i], alists[i]), NZ)
#    coeffs = ntuple(i -> T.(D["coeffs_re"][i]) + im * T.(D["coeffs_im"][i]), NZ)
#    return  SHIP(J, SH, zlist, alists, aalists, coeffs)
# end


# ------------------------------------------------------------
#   Evaluation code
# ------------------------------------------------------------



alloc_temp(V::PIPotential{T}, maxN::Integer) where {T} =
   (
      R = zeros(JVecF, maxN),
      Z = zeros(AtomicNumber, maxN),
      tmp_pibasis = alloc_temp(V.pibasis, maxN),
  )



# compute one site energy
function evaluate!(tmp, V::PIPotential,
                   Rs::AbstractVector{JVec{T}},
                   Zs::AbstractVector{<:AtomicNumber},
                   z0::AtomicNumber) where {T}
   iz0 = z2i(V, z0)
   A = evaluate!(tmp.tmp_pibasis.A, tmp.tmp_pibasis.tmp_basis1p,
                 V.pibasis.basis1p, Rs, Zs, z0)
   inner = V.pibasis.inner[iz0]
   c = V.coeffs[iz0]
   Es = zero(T)
   for iAA = 1:length(inner)
      Esi = c[iAA] # one(Complex{T})
      for α = 1:inner.orders[iAA]
         Esi *= A[inner.iAA2iA[iAA, α]]
      end
      Es += real(Esi)
   end
   return Es
end


# alloc_temp_d(ship::SHIP{T, NZ}, N::Integer) where {T, NZ} =
#       ( J = alloc_B(ship.J),
#        dJ = alloc_dB(ship.J),
#         Y = alloc_B(ship.SH),
#        dY = alloc_dB(ship.SH),
#         A = [ zeros(Complex{T}, length(ship.alists[iz])) for iz=1:NZ ],
#      dAco = [ zeros(Complex{T}, length(ship.alists[iz])) for iz=1:NZ ],
#      tmpJ = alloc_temp_d(ship.J),
#      tmpY = alloc_temp_d(ship.SH),
#        dV = zeros(JVec{T}, N),
#         R = zeros(JVec{T}, N),
#         Z = zeros(Int16, N)
#       )
#
# # compute one site energy
# function evaluate_d!(dEs, tmp, ship::SHIP{T},
#                      Rs::AbstractVector{JVec{T}},
#                      Zs::AbstractVector{<:Integer},
#                      z0::Integer
#                      ) where {T}
#    iz0 = z2i(ship, z0)
#    alist = ship.alists[iz0]
#    aalist = ship.aalists[iz0]
#
#    # stage 1: precompute all the A values
#    precompute_A!(tmp.A[iz0], tmp, alist, Rs, Zs, ship)
#
#    # stage 2: compute the coefficients for the ∇A_{klm} = ∇ϕ_{klm}
#    A = tmp.A[iz0]
#    dAco = tmp.dAco[iz0]
#    _evaluate_d_stage2!(dAco, A, aalist, ship.coeffs[iz0], ship)
#
#    # stage 3: get the gradients
#    fill!(dEs, zero(JVec{T}))
#    for (iR, (R, Z)) in enumerate(zip(Rs, Zs))
#       R̂ = R / norm(R)
#       evaluate_d!(tmp.J, tmp.dJ, tmp.tmpJ, ship.J, norm(R))
#       evaluate_d!(tmp.Y, tmp.dY, tmp.tmpY, ship.SH, R)
#       iz = z2i(ship, Z)
#       for iA = alist.firstz[iz]:(alist.firstz[iz+1]-1)
#          zklm = alist[iA]
#          ik, iy = zklm.k+1, index_y(zklm.l, zklm.m)
#          dEs[iR] += real(dAco[iA] * (
#                          tmp.J[ik] * tmp.dY[iy] + tmp.dJ[ik] * tmp.Y[iy] * R̂) )
#       end
#    end
#    return dEs
# end
#
# function _evaluate_d_stage2!(dAco::AbstractVector{CT}, A, aalist, c, ship::SHIP
#                             ) where {CT}
#    fill!(dAco, 0)
#    for iAA = 1:length(aalist)
#       for α = 1:aalist.len[iAA]
#          CxA_α = CT(c[iAA])
#          for β = 1:aalist.len[iAA]
#             if β != α
#                iAβ = aalist.i2Aidx[iAA, β]
#                CxA_α *= A[iAβ]
#             end
#          end
#          iAα = aalist.i2Aidx[iAA, α]
#          dAco[iAα] += CxA_α
#       end
#    end
# end
