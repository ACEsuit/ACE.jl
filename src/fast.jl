
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


using SHIPs.SphericalHarmonics: SHBasis, index_y
using StaticArrays
using JuLIP: AbstractCalculator, Atoms, JVec, SitePotential
using NeighbourLists: max_neigs, neigs

import JuLIP, JuLIP.MLIPs
import JuLIP: cutoff, alloc_temp, alloc_temp_d
import JuLIP.Potentials: evaluate!, evaluate_d!
import JuLIP.MLIPs: combine
import Base: Dict, convert, ==

export SHIP

struct SHIP{BO, T, TJ} <: SitePotential
   J::TJ
   SH::SHBasis{T}
   KL::Vector{NamedTuple{(:k, :l, :deg),Tuple{Int,Int,T}}}
   firstA::Vector{Int}      # indexing into A
   # --------------
   IA::Vector{SVector{BO,Int}}
   C::Vector{T}
end

cutoff(ship::SHIP) = cutoff(ship.J)

==(S1::SHIP, S2::SHIP) = (
      (bodyorder(S1) == bodyorder(S2)) &&
      (S1.J == S2.J) &&
      (S1.SH == S2.SH) &&
      (S1.KL == S2.KL) &&
      (S1.firstA == S2.firstA) &&
      (S1.IA == S2.IA) &&
      (S1.C == S2.C) )

Dict(ship::SHIP) = Dict(
      "__id__" => "SHIPs_SHIP",
      "bodyorder" => bodyorder(ship),
      "J" => Dict(ship.J),
      "SH_maxL" => ship.SH.maxL,
      "T" => string(eltype(ship.SH)),
      "K" => [kl.k for kl in ship.KL],
      "L" => [kl.l for kl in ship.KL],
      "KLD" => [kl.deg for kl in ship.KL],
      "firstA" => ship.firstA,
      "IA" => ship.IA,
      "C" => ship.C
   )

# bodyorder - 1 is because BO is the number of neighbours
# not the actual body-order
SHIP(D::Dict) = _SHIP(D, Val(Int(D["bodyorder"]-1)),
                         Meta.eval(Meta.parse(D["T"])))

_SHIP(D::Dict, ::Val{BO}, T) where {BO} = SHIP(
      TransformedJacobi(D["J"]),
      SHBasis(D["SH_maxL"], T),
      [ (k = k, l = l, deg = T(deg)) for (k, l, deg) in zip(D["K"], D["L"], D["KLD"]) ],
      Vector{Int}(D["firstA"]),
      Vector{SVector{BO,Int}}(D["IA"]),
      Vector{T}(D["C"])
   )

convert(::Val{:SHIPs_SHIP}, D::Dict) = SHIP(D)

Base.length(ship::SHIP) = length(ship.C)

# BO + 1 because BO is the number of neighbours not the actual body-order
bodyorder(ship::SHIP{BO}) where {BO} = BO + 1

length_A(ship::SHIP) = ship.firstA[end]

combine(basis::SHIPBasis, coeffs) = SHIP(basis, coeffs)

function SHIP(basis::SHIPBasis{BO, T}, coeffs::AbstractVector{T}) where {BO, T}
   IA = SVector{BO,Int}[]
   C = T[]
   ia = zero(MVector{BO, Int})
   for (idx, ν) in enumerate(basis.Nu)
      kk, ll, mrange = _klm(ν, basis.KL)
      for mpre in mrange
         mm = SVector(Tuple(mpre)..., - sum(Tuple(mpre)))
         # skip any m-tuples that aren't admissible
         if abs(mm[end]) > ll[end]; continue; end
         # compute the coefficient of a ∏ Aⱼ term
         c = _Bcoeff(ll, mm, basis.cg) * coeffs[idx]
         push!(C, c)
         # compute the indices of Aⱼ in the store.A array
         for i = 1:BO
            ia[i] = basis.firstA[ν[i]] + ll[i] + mm[i]
         end
         push!(IA, SVector(ia))
      end
   end
   return SHIP( basis.J, basis.SH, copy(basis.KL), copy(basis.firstA), IA, C)
end



alloc_temp(ship::SHIP{BO,T}, ::Integer) where {BO, T} =
   (     J = alloc_B(ship.J),
         Y = alloc_B(ship.SH),
         A = zeros(Complex{T}, length_A(ship)),
      tmpJ = alloc_temp(ship.J),
      tmpY = alloc_temp(ship.SH)
   )


function precompute!(store, ship::SHIP, Rs)
   fill!(store.A, 0.0)
   for (iR, R) in enumerate(Rs)
      # evaluate the r-basis and the R̂-basis for the current neighbour at R
      eval_basis!(store.J, ship.J, norm(R), store.tmpJ)
      eval_basis!(store.Y, ship.SH, R, store.tmpY)
      # add the contributions to the A_klm; the indexing into the
      # A array is determined by `ship.firstA` which was precomputed
      for ((k, l), iA) in zip(ship.KL, ship.firstA)
         for m = -l:l
            store.A[iA+l+m] += store.J[k+1] * store.Y[index_y(l, m)]
         end
      end
   end
   return nothing
end

# compute one site energy
function evaluate!(store, ship::SHIP{BO, T}, Rs::AbstractVector{JVec{T}}) where {BO, T}
   precompute!(store, ship, Rs)
   Es = T(0.0)
   for (iA, c) in zip(ship.IA, ship.C)
      @inbounds Es_ν = Complex{T}(c) * prod(store.A[iA])
      Es += real(Es_ν)
   end
   return Es
end

alloc_temp_d(ship::SHIP{BO, T}, N::Integer) where {BO, T} =
      ( J = alloc_B(ship.J),
       dJ = alloc_dB(ship.J),
        Y = alloc_B(ship.SH),
       dY = alloc_dB(ship.SH),
        A = zeros(Complex{T}, length_A(ship)),
     dAco = zeros(Complex{T}, length_A(ship)),
     tmpJ = alloc_temp_d(ship.J),
     tmpY = alloc_temp_d(ship.SH),
       dV = zeros(JVec{T}, N)
      )

# compute one site energy
function evaluate_d!(dEs, store, ship::SHIP{BO, T}, Rs::AbstractVector{JVec{T}},
                     ) where {BO, T}

   # stage 1: precompute all the A values
   precompute!(store, ship, Rs)

   # stage 2: compute the coefficients for the ∇A_{klm}
   #          (and also Es while we are at it)
   fill!(store.dAco, T(0))
   Es = T(0)
   for (iA, c) in zip(ship.IA, ship.C)
      # compute the coefficients
      for α = 1:BO
         CxA_α = Complex{T}(c)
         for β = 1:BO
            if β != α
               @inbounds CxA_α *= store.A[iA[β]]
            end
         end
         @inbounds store.dAco[iA[α]] += CxA_α
      end
   end

   # stage 3: get the gradients
   fill!(dEs, zero(JVec{T}))

   for (iR, R) in enumerate(Rs)
      eval_basis_d!(store.J, store.dJ, ship.J, norm(R), store.tmpJ)
      eval_basis_d!(store.Y, store.dY, ship.SH, R, store.tmpY)
      for ((k, l), iA) in zip(ship.KL, ship.firstA)
         for m = -l:l
            @inbounds aaa = store.J[k+1] * store.dY[index_y(l, m)]
            @inbounds bbb = (store.dJ[k+1] * store.Y[index_y(l, m)]) * (R/norm(R))
            @inbounds dEs[iR] += real( store.dAco[iA+l+m] * (
                  store.J[k+1] * store.dY[index_y(l, m)] +
                  (store.dJ[k+1] * store.Y[index_y(l, m)]) * (R/norm(R)) ) )
         end
      end
   end

   return dEs
end





# [ Info: profile basis and ship computation
# [ Info:   body-order 3:
# [ Info:      evaluate a site energy:
#          SHIPBasis:   189.277 μs (0 allocations: 0 bytes)
#          SHIP     :   310.281 μs (0 allocations: 0 bytes)
# [ Info:      site energy gradient:
#          SHIPBasis:   4.055 ms (450 allocations: 1.39 MiB)
#          SHIP     :   1.223 ms (0 allocations: 0 bytes)
# [ Info:   body-order 4:
# [ Info:      evaluate a site energy:
#          SHIPBasis:   599.627 μs (0 allocations: 0 bytes)
#          SHIP     :   262.199 μs (0 allocations: 0 bytes)
# [ Info:      site energy gradient:
#          SHIPBasis:   57.463 ms (400 allocations: 659.38 KiB)
#          SHIP     :   768.099 μs (0 allocations: 0 bytes)
# [ Info:   body-order 5:
# [ Info:      evaluate a site energy:
#          SHIPBasis:   1.006 ms (0 allocations: 0 bytes)
#          SHIP     :   185.435 μs (0 allocations: 0 bytes)
# [ Info:      site energy gradient:
#          SHIPBasis:   56.504 ms (400 allocations: 419.53 KiB)
#          SHIP     :   623.716 μs (0 allocations: 0 bytes)
