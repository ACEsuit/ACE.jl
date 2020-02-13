
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


using PoSH: alloc_temp, alloc_B, evaluate!
using LinearAlgebra: dot
using PoSH.OrthPolys: OrthPolyBasis

import PoSH: alloc_B, alloc_dB


struct OrthPolyBasis1{T}   # T = Float64
   # ...
   # ???
   # ...
   uselast::Bool
   # ----------------- used only for construction ...
   #                   but useful to have since it defines the notion or orth.
   tdf::Vector{T}
   ww::Vector{T}
end


Base.length(P::OrthPolyBasis1) = length(P.A) - 1 + P.uselast

function Base.rand(J::OrthPolyBasis1)
   @assert maximum(abs, diff(J.ww)) == 0
   return rand(J.tdf)
end

# SIMON : construct the new basis
function OrthPolyBasis1(P::OrthPolyBasis)

   # inner product: fi = fi.(P.tdf)
   dotw = (f1, f2) -> dot(f1, P.ww .* f2)

   # evaluate the basis at one point
   tmp = alloc_temp(P)
   p = alloc_B(P)
   evalP = t -> evaluate!(p, tmp, P, t)
   # returns p::Vector{T} containing the values of the basis functions at t::T
   # or use
   evaluate(P, t)

   # output: OrthPolyBasis1
end

alloc_B( J::OrthPolyBasis1{T}) where {T} = zeros(T, length(J))
alloc_dB(J::OrthPolyBasis1{T}) where {T} = zeros(T, length(J))

alloc_B( J::OrthPolyBasis1{T}, x::TX) where {T, TX} = zeros(TX, length(J))
alloc_dB(J::OrthPolyBasis1{T}, x::TX) where {T, TX} = zeros(TX, length(J))

# SIMON : evaluate the new basis
function evaluate!(P::AbstractVector{T}, tmp::Nothing,
                   J::OrthPolyBasis1{T}, t::T) where {T}
   

   return P
end
