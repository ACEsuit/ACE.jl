
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


using SHIPs.SphericalHarmonics: SHBasis, index_y
using StaticArrays
using JuLIP: AbstractCalculator, Atoms, JVec, SitePotential

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

Base.length(ship::SHIP) = length(ship.C)

length_A(ship::SHIP) = ship.firstA[end]


function SHIP(basis::SHIPBasis{BO, T}, coeffs::Vector{T}) where {BO, T}
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



alloc_temp(ship::SHIP{BO,T}) where {BO, T} =
   (  J = alloc_B(ship.J),
      Y = alloc_B(ship.SH),
      A = zeros(Complex{T}, length_A(ship))
   )


function precompute!(store, ship::SHIP, Rs)
   fill!(store.A, 0.0)
   for (iR, R) in enumerate(Rs)
      # evaluate the r-basis and the R̂-basis for the current neighbour at R
      eval_basis!(store.J, ship.J, norm(R), nothing)
      eval_basis!(store.Y, ship.SH, R, nothing)
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
function evaluate!(ship::SHIP{BO, T}, Rs::AbstractVector{JVec{T}}, store) where {BO, T}
   precompute!(store, ship, Rs)
   Es = 0.0
   for (iA, c) in zip(ship.IA, ship.C)
      @inbounds Es_ν = Complex{T}(c) * prod(store.A[iA])
      Es += real(Es_ν)
   end
   return Es
end

alloc_temp_d(ship::SHIP{BO, T}, Rs::AbstractVector{<:SVector}) where {BO, T} =
      alloc_temp_d(ship, length(Rs))

alloc_temp_d(ship::SHIP{BO, T}, N::Integer) where {BO, T} =
      ( J = alloc_B(ship.J),
       dJ = alloc_dB(ship.J),
        Y = alloc_B(ship.SH),
       dY = alloc_dB(ship.SH),
        A = zeros(Complex{T}, length_A(ship)),
     dAco = zeros(Complex{T}, length_A(ship))
      )

# compute one site energy
function evaluate_d!(dEs, ship::SHIP{BO, T}, Rs::AbstractVector{JVec{T}},
                     store) where {BO, T}

   # stage 1: precompute all the A values
   precompute!(store, ship, Rs)

   # stage 2: compute the coefficients for the ∇A_{klm}
   #          (and also Es while we are at it)
   fill!(store.dAco, 0.0)
   Es = 0.0
   for (iA, c) in zip(ship.IA, ship.C)
      @inbounds Es_ν = Complex{T}(c) * prod(store.A[iA])
      Es += real(Es_ν)
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
      eval_basis_d!(store.J, store.dJ, ship.J, norm(R), nothing)
      eval_basis_d!(store.Y, store.dY, ship.SH, R, nothing)
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

   return Es
end


# ------------ JuLIP Calculators ------------------
#  * forces and virials should just follow from JuLIP.

function energy(at::Atoms, ship::SHIP)

end 
