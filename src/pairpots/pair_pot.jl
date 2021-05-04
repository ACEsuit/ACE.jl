
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------



# ----------------------------------------------------

export PolyPairPot

struct PolyPairPot{T,TJ,NZ} <: PairPotential
   coeffs::Vector{T}
   basis::PolyPairBasis{TJ, NZ}
end

@pot PolyPairPot

PolyPairPot(pB::PolyPairBasis, coeffs::Vector) = PolyPairPot(coeffs, pB)

JuLIP.MLIPs.combine(pB::PolyPairBasis, coeffs::AbstractVector) =
            PolyPairPot(identity.(collect(coeffs)), pB)

JuLIP.cutoff(V::PolyPairPot) = cutoff(V.basis)

z2i(V::PolyPairPot, z::AtomicNumber) = z2i(V.basis, z)
i2z(V::PolyPairPot, iz::Integer) = i2z(V.basis, iz)
numz(V::PolyPairPot) = numz(V.basis)

==(V1::PolyPairPot, V2::PolyPairPot) =
            ( (V1.basis == V2.basis) && (V1.coeffs == V2.coeffs) )

write_dict(V::PolyPairPot{T}) where {T} = Dict(
      "__id__" => "ACE_PolyPairPot",
      "T" => write_dict(T),
      "coeffs" => V.coeffs,
      "basis" => write_dict(V.basis)
      )

read_dict(::Val{:SHIPs_PolyPairPot}, D::Dict) =
   read_dict(Val{:ACE_PolyPairPot}(), D)

read_dict(::Val{:ACE_PolyPairPot}, D::Dict, T = read_dict(D["T"])) =
      PolyPairPot(read_dict(D["basis"]), T.(D["coeffs"]))


alloc_temp(V::PolyPairPot{T}, N::Integer) where {T} =
      ( R = zeros(JVec{T}, N),
        Z = zeros(AtomicNumber, N),
        alloc_temp(V.basis)... )

alloc_temp_d(V::PolyPairPot{T}, N::Integer) where {T} =
      ( dV = zeros(JVec{T}, N),
         R = zeros(JVec{T}, N),
         Z = zeros(AtomicNumber, N),
         alloc_temp_d(V.basis)... )


function _dot_zij(V, B, z, z0)
   i0 = _Bidx0(V.basis, z, z0)  # cf. pair_basis.jl
   return sum( V.coeffs[i0 + n] * B[n]  for n = 1:length(V.basis, z0) )
end

evaluate!(tmp, V::PolyPairPot, r::Number, z, z0) =
      _dot_zij(V, evaluate!(tmp.J, tmp.tmp_J, V.basis.J, r), z, z0)

evaluate_d!(tmp, V::PolyPairPot, r::Number, z, z0) =
      _dot_zij(V, evaluate_d!(tmp.J, tmp.dJ, tmp.tmpd_J, V.basis.J, r), z, z0)

function evaluate!(tmp, V::PolyPairPot, r::Number)
   @assert numz(V) == 1
   z = i2z(V, 1)
   return evaluate!(tmp, V::PolyPairPot, r::Number, z, z)
end

function evaluate_d!(tmp, V::PolyPairPot, r::Number)
   @assert numz(V) == 1
   z = i2z(V, 1)
   return evaluate_d!(tmp, V::PolyPairPot, r::Number, z, z)
end

evaluate(V::PolyPairPot, r::Number) = evaluate!(alloc_temp(V, 1), V, r)
evaluate_d(V::PolyPairPot, r::Number) = evaluate_d!(alloc_temp_d(V, 1), V, r)
