
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


# k : r-degree
# l : θ-degree
# m : z-degree

import SHIPs: alloc_B, alloc_dB

import JuLIP: evaluate!, evaluate_d!,
              alloc_temp, alloc_temp_d,
              read_dict, write_dict

import JuLIP.MLIPs: IPBasis


struct EnvPairBasis{T0, TR, TZ, TT, TI} <: IPBasis
   P0::T0                # basis for the bond-length coordinate / m0 = k0
   Pr::TR                # specifies the radial basis  / n = kr
   Pθ::TT                # the angular basis           / l = kθ
   Pz::TZ                # specifies the z-basis       / m = kz
   aalist::BondAAList{TI}    # datastructure specifying the basis
end

Base.length(basis::EnvPairBasis) = length(basis.aalist)

Base.eltype(basis::EnvPairBasis) = Complex{Float64}   # eltype(basis.Pr)

alloc_B(basis::EnvPairBasis, args...) = zeros(eltype(basis), length(basis))

alloc_temp(basis::EnvPairBasis, args...) =
   ( P0 = alloc_B(basis.P0),
     tmp_P0 = alloc_temp(basis.P0),
     Pr = alloc_B(basis.Pr),
     tmp_Pr = alloc_temp(basis.Pr),
     Pθ = alloc_B(basis.Pθ),
     tmp_Pθ = alloc_temp(basis.Pθ),
     Pz = alloc_B(basis.Pz),
     tmp_Pz = alloc_temp(basis.Pz),
     A = alloc_A(basis.aalist.alist)
    )

function precompute_A!(A, tmp, basis::EnvPairBasis, R0, Rs)
   alist = basis.aalist.alist
   # construct the coordinate system, and convert
   cylcoords = CylindricalCoordinateSystem(R0)
   # loop through the environment to assemble the As
   for R in Rs
      rθz = cylcoords(R)
      evaluate!(tmp.Pr, tmp.tmp_Pr, basis.Pr, rθz.r)
      evaluate!(tmp.Pz, tmp.tmp_Pz, basis.Pz, rθz.z)
      evaluate!(tmp.Pθ, tmp.tmp_Pθ, basis.Pθ, rθz)
      for i = 1:length(alist)
         krθz = alist[i]
         A[i] += tmp.Pr[krθz.kr+1] *
                 tmp.Pθ[cyl_l2i(krθz.kθ, basis.Pθ)] *
                 tmp.Pz[krθz.kz+1]
      end
      # iz = z2i(ship, Z)
      # for i = alist.firstz[iz]:(alist.firstz[iz+1]-1)
      #    zklm = alist[i]
      #    A[i] += tmp.Pr[zklm.k+1] * tmp.Pθ[cyl_l2i(zklm.l)] * tmp.Pz[zklm.m+1]
      # end
   end
   return A
end

# R0   : typically SVector{T}
# Renv : typically Vector{SVector{T}} or a view into Vector{SVector{T}}
function evaluate!(B::AbstractVector{Complex{T}},
                   tmp, basis::EnvPairBasis, R0, Renv)  where {T}
   aalist = basis.aalist
   # construct the basis for the r1-variable
   r0 = norm(R0)
   P0 = evaluate!(tmp.P0, tmp.tmp_P0, basis.P0, r0)

   # evaluate the A-basis, i.e. the density projections of the environment
   A = precompute_A!(tmp.A, tmp, basis, R0, Renv)

   # loop over all basis functions
   for i = 1:length(aalist)
      B[i] = P0[aalist.i2k0[i] + 1]
      for α = 1:aalist.len[i]
         B[i] *= A[aalist.i2Aidx[i, α]]  # TODO: reverse i, α
      end
   end

   return B
end
