

import JuLIP
using JuLIP: JVec, Atoms
using JuLIP.Potentials: @pot, MPairPotential, SZList, ZList, z2i, i2z
using LinearAlgebra: dot

import JuLIP.Potentials: evaluate, evaluate_d

# ----------------------------------------------------

export PolyPairPot

struct PolyPairPot{T,TJ,NZ} <: MPairPotential
   coeffs::Vector{T}
   J::TJ
   zlist::SZList{NZ}
   bidx0::SMatrix{NZ,NZ,Int16}
end

@pot PolyPairPot

PolyPairPot(pB::PolyPairBasis, coeffs::Vector) =
            PolyPairPot(coeffs, pB.J, pB.zlist, pB.bidx0)

JuLIP.MLIPs.combine(pB::PolyPairBasis, coeffs::AbstractVector) =
            PolyPairPot(pB, collect(coeffs))

JuLIP.cutoff(V::PolyPairPot) = cutoff(V.J)

==(V1::PolyPairPot, V2::PolyPairPot) =
            ( (V1.J == V2.J) && (V1.coeffs == V2.coeffs) &&  V1.zlist == V2.zlist )

Dict(V::PolyPairPot) = Dict(
      "__id__" => "PolyPairPots_PolyPairPot",
      "coeffs" => V.coeffs,
      "J" => Dict(V.J),
      "zlist" => V.zlist.list
      )

function PolyPairPot(D::Dict)
   J = TransformedJacobi(D["J"])
   zlist = ZList(Int16.(D["zlist"]), static = true)
   return PolyPairPot( Vector{Float64}(D["coeffs"]),
                       J, zlist, get_bidx0(J, zlist) )
end

convert(::Val{:PolyPairPots_PolyPairPot}, D::Dict) = PolyPairPot(D)


alloc_temp(V::PolyPairPot{T}, N::Integer) where {T} =
      ( J = alloc_B(V.J),
        R = zeros(JVec{T}, N),
        Z = zeros(Int16, N) )

alloc_temp_d(V::PolyPairPot{T}, N::Integer) where {T} =
      ( J = alloc_B(V.J),
       dJ = alloc_dB(V.J),
       dV = zeros(JVec{T}, N),
        R = zeros(JVec{T}, N),
        Z = zeros(Int16, N) )


function _dot_zij(V, B, z, z0)
   i0 = _Bidx0(V, z, z0)
   return sum( V.coeffs[i0 + n] * B[n]  for n = 1:length(V.J) )
end

evaluate!(tmp, V::PolyPairPot, r::Number, z, z0) =
      _dot_zij(V, evaluate!(tmp.J, nothing, V.J, r), z, z0)

evaluate_d!(tmp, V::PolyPairPot, r::Number, z, z0) =
      _dot_zij(V, evaluate_d!(tmp.J, tmp.dJ, nothing, V.J, r), z, z0)

function evaluate!(tmp, V::PolyPairPot, r::Number)
   @assert length(V.zlist) == 1
   z = V.zlist.list[1]
   return evaluate!(tmp, V::PolyPairPot, r::Number, z, z)
end

function evaluate_d!(tmp, V::PolyPairPot, r::Number)
   @assert length(V.zlist) == 1
   z = V.zlist.list[1]
   return evaluate_d!(tmp, V::PolyPairPot, r::Number, z, z)
end

evaluate(V::PolyPairPot, r::Number) = evaluate!(alloc_temp(V, 1), V, r)
evaluate_d(V::PolyPairPot, r::Number) = evaluate_d!(alloc_temp_d(V, 1), V, r)
