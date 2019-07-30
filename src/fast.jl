
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


using SHIPs.SphericalHarmonics: SHBasis, index_y
using StaticArrays
using JuLIP: AbstractCalculator, Atoms, JVec
using JuLIP.Potentials: MSitePotential
using NeighbourLists: max_neigs, neigs

import JuLIP, JuLIP.MLIPs
import JuLIP: cutoff, alloc_temp, alloc_temp_d
import JuLIP.Potentials: evaluate!, evaluate_d!
import JuLIP.MLIPs: combine
import Base: Dict, convert, ==

export SHIP

struct SHIP{BO, T, NZ, TJ} <: MSitePotential
   J::TJ
   SH::SHBasis{T}
   # -------------- 1-particle basis
   KL::NTuple{NZ, Vector{NamedTuple{(:k, :l),Tuple{IntS,IntS}}}}    # 1-particle indexing
   firstA::NTuple{NZ, Vector{IntS}}   # indexing into A-basis vectors
   # -------------- n-particle basis
   spec::SMatrix{BO, NZ, Vector}
   # IA::SMatrix{BO, NZ, Vector}        # IA[n, iz]::Vector{SVector{N, IntS}}
   # ZA::SMatrix{BO, NZ, Vector}        # ZA[n, iz]::Vector{SVector{N, Int16}}
   # C::SMatrix{BO, NZ, Vector{T}}      # sub-coefficients
   # --------------
   i2z::NTuple{NZ, Int16}
end


cutoff(ship::SHIP) = cutoff(ship.J)

i2z(ship::SHIP, iz::Integer) = ship.i2z[iz]

function z2i(ship::SHIP, z::Integer)
   for iz = 1:length(ship.i2z)
      if z == ship.i2z[iz]
         return iz
      end
   end
   error("Atom number `$z` was not found.")
end


==(S1::SHIP, S2::SHIP) =
      all( getfield(S1, i) == getfield(S2, i)
           for i = 1:fieldcount(SparseSHIP) )

# ==(S1::SHIP, S2::SHIP) = (
#       (bodyorder(S1) == bodyorder(S2)) &&
#       (S1.J == S2.J) &&
#       (S1.SH == S2.SH) &&
#       (S1.KL == S2.KL) &&
#       (S1.firstA == S2.firstA) &&
#       (S1.IA == S2.IA) &&
#       (S1.C == S2.C) )


Base.length(ship::SHIP) = sum(length.(ship.spec))

# BO + 1 because BO is the number of neighbours not the actual body-order
bodyorder(ship::SHIP{BO}) where {BO} = BO + 1


# ------------------------------------------------------------
#   Initialisation code
# ------------------------------------------------------------

combine(basis::SHIPBasis, coeffs) = SHIP(basis, coeffs)

const Tspec{N, T, TZ, TI} = NamedTuple{(:izz, :iA, :c),
                                     Tuple{SVector{N, TZ},
                                           SVector{N, TI},
                                           T}
                                    }

function _init_spec(bo::Integer, nz::Integer, T=Float64, TI=IntS, TZ=Int16)
   spec = Matrix{Vector}(undef, bo, nz)
   for n = 1:bo, iz = 1:nz
      spec[n, iz] = Tspec{n, T, TZ, TI}[]
   end
   return SMatrix{bo, nz, Vector}(spec)
end


function SHIP(basis::SHIPBasis{BO, T, NZ}, coeffs::AbstractVector{T}
              ) where {BO, T, NZ}
   spec = _init_spec(BO, NZ, T)
   for N = 1:BO, iz = 1:NZ
      _get_C_IA!(spec[N, iz], basis, coeffs, Val(N), iz)
   end
   #  TODO: basis.spec.Zs => this is a hack. Store this in `SHIPBasis`
   #        and make it available via an interface
   return SHIP( basis.J, basis.SH,
                deepcopy(basis.KL), deepcopy(basis.firstA),
                spec,
                basis.spec.Zs )
end

function _get_C_IA!(spec, basis, coeffs, ::Val{N}, iz0) where {N}
   ia = zero(MVector{N, IntS})
   NuZ_N = basis.NuZ[N, iz0]::Vector{Tνz{N}}
   idx0 = _first_B_idx(basis, N, iz0)
   for (idx, νz) in enumerate(NuZ_N)
      ν = νz.ν
      izz = νz.izz
      idxB = idx0 + idx
      kk, ll = _kl(ν, basis.KL[izz])   # TODO: allocation -> fix this!
      for mm in _mrange(ll)
         # skip any m-tuples that aren't admissible:
         # TODO: incorporate this into _mrange
         if abs(mm[end]) > ll[end]; continue; end
         # compute the coefficient of a ∏ Aⱼ term
         c = _Bcoeff(ll, mm, basis.cg) * coeffs[idxB]
         # compute the indices of Aⱼ in the store.A array
         for α = 1:N
            ia[α] = basis.firstA[izz[α]][ν[α]] + ll[α] + mm[α]
         end
         push!(spec, ( izz=izz, iA = SVector(ia), c = c ))
      end
   end
   return nothing
end


# ------------------------------------------------------------
#   FIO code
# ------------------------------------------------------------

function _spec2dict(spec::SMatrix{BO, NZ}) where {BO, NZ}
   specD = []
   for iz = 1:NZ, N = 1:BO
      specD_izN = []
      for s in spec[N, iz]
         push!(specD_izN, [s.izz, s.iA, s.c])
      end
      push!(specD, specD_izN)
   end
   return specD
end

function _dict2spec(specD, bo, nz)
   spec = _init_spec(bo, nz)
   idx = 0
   for iz = 1:nz, N = 1:bo
      idx += 1
      specD_izN = specD[idx]
      for sD in specD_izN
         izz = SVector(Int16.(sD[1]))
         iA = SVector(IntS.(sD[2]))
         c = Float64(sD[3])
         push!(spec[N, iz], (izz=izz, iA=iA, c=c))
      end
   end
   return spec
end

Dict(ship::SHIP{BO,T,NZ}) where {BO, T, NZ} = Dict(
      "__id__" => "SHIPs_SHIP",
      "bodyorder" => bodyorder(ship),
      "J" => Dict(ship.J),
      "SH_maxL" => ship.SH.maxL,   # TODO: replace this with Dict(SH)
      "T" => string(eltype(ship.SH)),
      "K" => [ [kl.k for kl in ship.KL[iz]] for iz = 1:NZ ],
      "L" => [ [kl.l for kl in ship.KL[iz]] for iz = 1:NZ ],
      "firstA" => ship.firstA[:],
      "spec" => _spec2dict(ship.spec),
      "Z" => ship.i2z
   )

# bodyorder - 1 is because BO is the number of neighbours
# not the actual body-order
SHIP(D::Dict) = _SHIP(D, Val(Int(D["bodyorder"]-1)),
                         Meta.eval(Meta.parse(D["T"])),
                         Val(Int(length(D["Z"]))) )

function _SHIP(D::Dict, ::Val{BO}, T, ::Val{NZ}) where {BO, NZ}
   spec = _dict2spec(D["spec"], BO, NZ)
   KL =  [ [ (k = k, l = l) for (k, l) in zip(D["K"][iz], D["L"][iz]) ]
           for iz = 1:NZ ]
   firstA = [ Vector{IntS}(D["firstA"][iz]) for iz = 1:NZ ]

   return  SHIP(
      TransformedJacobi(D["J"]),
      SHBasis(D["SH_maxL"], T),
      tuple(KL...),
      tuple(firstA...),
      spec,
      Int16.(D["Z"]) )
end

convert(::Val{:SHIPs_SHIP}, D::Dict) = SHIP(D)


# ------------------------------------------------------------
#   Evaluation code
# ------------------------------------------------------------

length_A(ship::SHIP, iz) = ship.firstA[iz][end]


alloc_temp(ship::SHIP{BO,T,NZ}, N::Integer) where {BO, T, NZ} =
   (     J = alloc_B(ship.J),
         Y = alloc_B(ship.SH),
         A = [ zeros(Complex{T}, length_A(ship, iz)) for iz=1:NZ ],
      tmpJ = alloc_temp(ship.J),
      tmpY = alloc_temp(ship.SH),
         R = zeros(JVec{T}, N),
         Z = zeros(Int16, N)
           )


function precompute!(tmp, ship::SHIP, Rs, Zs)
   _zero_A!(tmp.A)
   for (iR, (R, Z)) in enumerate(zip(Rs, Zs))
      # evaluate the r-basis and the R̂-basis for the current neighbour at R
      eval_basis!(tmp.J, tmp.tmpJ, ship.J, norm(R))
      eval_basis!(tmp.Y, tmp.tmpY, ship.SH, R)
      # add the contributions to the A_klm; the indexing into the
      # A array is determined by `ship.firstA` which was precomputed
      iz = z2i(ship, Z)
      for ((k, l), iA) in zip(ship.KL[iz], ship.firstA[iz])
         for m = -l:l
            tmp.A[iz][iA+l+m] += tmp.J[k+1] * tmp.Y[index_y(l, m)]
         end
      end
   end
   return nothing
end

# compute one site energy
function evaluate!(tmp, ship::SHIP{BO, T},
                   Rs::AbstractVector{JVec{T}},
                   Zs::AbstractVector{<:Integer},
                   z0::Integer) where {BO, T}
   iz0 = z2i(ship, z0)
   precompute!(tmp, ship, Rs, Zs)
   return valnmapreduce(Val(BO), T(0.0),
                        valN -> _evaluate!(tmp, ship, iz0, valN))
end

function _evaluate!(tmp, ship::SHIP{BO, T, NZ}, iz0, ::Val{N}) where {BO, T, N, NZ}
   Es = T(0.0)
   spec_N = ship.spec[N, iz0]::Vector{Tspec{N, T, Int16, IntS}}
   # IA_N = ship.IA[N, iz]::Vector{SVector{N, IntS}}
   # C_N = ship.C[N]::Vector{T}
   for s in spec_N  # (iA, c) in zip(IA_N, C_N)
      Es_ν = s.c
      for α = 1:length(s.iA)
         Es_ν *= tmp.A[s.izz[α]][s.iA[α]]
      end
      Es += real(Es_ν)
   end
   return Es
end


alloc_temp_d(ship::SHIP{BO, T, NZ}, N::Integer) where {BO, T, NZ} =
      ( J = alloc_B(ship.J),
       dJ = alloc_dB(ship.J),
        Y = alloc_B(ship.SH),
       dY = alloc_dB(ship.SH),
        A = [ zeros(Complex{T}, length_A(ship, iz)) for iz=1:NZ ],
     dAco = [ zeros(Complex{T}, length_A(ship, iz)) for iz=1:NZ ],
     tmpJ = alloc_temp_d(ship.J),
     tmpY = alloc_temp_d(ship.SH),
       dV = zeros(JVec{T}, N),
        R = zeros(JVec{T}, N),
        Z = zeros(Int16, N)
      )

# compute one site energy
function evaluate_d!(dEs, tmp, ship::SHIP{BO, T},
                     Rs::AbstractVector{JVec{T}},
                     Zs::AbstractVector{<:Integer},
                     z0::Integer
                     ) where {BO, T}
   iz0 = z2i(ship, z0)

   # stage 1: precompute all the A values
   precompute!(tmp, ship, Rs, Zs)

   # stage 2: compute the coefficients for the ∇A_{klm}
   _zero_A!(tmp.dAco)
   nfcalls(Val(BO), valN -> _evaluate_d_stage2!(tmp, ship, iz0, valN))

   # stage 3: get the gradients
   fill!(dEs, zero(JVec{T}))

   for (iR, (R, Z)) in enumerate(zip(Rs, Zs))
      eval_basis_d!(tmp.J, tmp.dJ, tmp.tmpJ, ship.J, norm(R))
      eval_basis_d!(tmp.Y, tmp.dY, tmp.tmpY, ship.SH, R)
      iz = z2i(ship, Z)
      for ((k, l), iA) in zip(ship.KL[iz], ship.firstA[iz])
         for m = -l:l
            @inbounds dEs[iR] += real( tmp.dAco[iz][iA+l+m] * (
                  tmp.J[k+1] * tmp.dY[index_y(l, m)] +
                  (tmp.dJ[k+1] * tmp.Y[index_y(l, m)]) * (R/norm(R)) ) )
         end
      end
   end

   return dEs
end

function _evaluate_d_stage2!(tmp, ship::SHIP{BO, T, NZ}, iz0, ::Val{N}
                             ) where {BO, T, N, NZ}
   spec_N = ship.spec[N, iz0]::Vector{Tspec{N, T, Int16, IntS}}

   # the site-energy assembly for comparison:
   # for s in spec_N
   #    Es_ν = s.c
   #    for α = 1:length(s.iA)
   #       Es_ν *= tmp.A[s.izz[α]][s.iA[α]]
   #    end
   #    Es += real(Es_ν)
   # end

   # for (iA, c) in zip(IA_N, C_N)
   for s in spec_N
      # compute the coefficients
      for α = 1:N
         CxA_α = Complex{T}(s.c)
         for β = 1:N
            if β != α
               @inbounds CxA_α *= tmp.A[s.izz[β]][s.iA[β]]
            end
         end
         @inbounds tmp.dAco[s.izz[α]][s.iA[α]] += CxA_α
      end
   end
end
