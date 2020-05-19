
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


module Tree

include("extimports.jl")
include("shipimports.jl")

using Combinatorics: combinations

import SHIPs: InnerPIBasis

struct TreePIPot{T, NZ, TB} <: SitePotential
   basis1p::TB
   trees::NTuple{NZ, Vector{Tuple{UInt16,UInt16}}}
   coeffs::NTuple{NZ, Vector{T}}
end

i2z(V::TreePIPot, i::Integer) = i2z(V.basis1p, i)
z2i(V::TreePIPot, z::AtomicNumber) = z2i(V.basis1p, z)
numz(V::TreePIPot) = numz(V.basis1p)

cutoff(V::TreePIPot) = cutoff(V.basis1p)

Base.eltype(V::TreePIPot{T}) where {T} = real(T)


# ---------------------------------------------------------------------
#   construction codes


function TreePIPot(pipot::PIPotential; kwargs...)
   trees = [ get_eval_tree(pipot.pibasis.inner[iz]; kwargs...)
             for iz = 1:numz(pipot) ]
   return TreePIPot(pipot.pibasis.basis1p, tuple(trees...), pipot.coeffs)
end




function get_eval_tree(inner::InnerPIBasis, filter = _->true)
   tree = Vector{Tuple{UInt16, UInt16}}(undef, length(inner))
   # make a list of all basis functions as vectors so we can search it
   spec = [ inner.iA2iAA[iAA, 1:inner.orders[iAA]]
            for iAA = 1:length(inner) ]
   # now look through all of them again
   for (ikk, kk) in enumerate(spec)
      # best length
      bestlen = length(kk)
      best_node = (UInt16(0), UInt16(0))
      for i1 in combinations(1:length(kk))
         if length(i1) > length(kk) ÷ 2; continue; end
         i2 = setdiff(1:length(kk), i1)
         kk1 = sort(kk[i1])
         kk2 = sort(kk[i2])
         len = max( length(kk1), length(kk2) )
         if len < bestlen
            n1 = findfirst(isequal(kk1), spec)
            n2 = findfirst(isequal(kk2), spec)
            if n1 != nothing && n2 != nothing
               bestlen = len
               best_node = (UInt16(n1), UInt16(n2))
            end
         end
      end
      if best_node == (UInt16(0), UInt16(0))
         error("couldn't find a decomposition for $kk")
      end
      tree[ikk] = best_node
   end
   return tree
end




# ---------------------------------------------------------------------
#   evaluation codes

alloc_temp(V::TreePIPot{T}, maxN::Integer) where {T} =
   (
   R = zeros(JVec{real(T)}, maxN),
   Z = zeros(AtomicNumber, maxN),
   tmp_basis1p = alloc_temp(V.basis1p),
   A = alloc_B(V.basis1p),
   AA = zeros(eltype(V.basis1p), length(V.coeffs))
    )


function evaluate!(tmp, V::TreePIPot, Rs, Zs, z0)
   iz0 = z2i(z0)
   AA = tmp.AA
   c = V.coeffs[iz0]
   tree = V.trees[iz0]

   # evaluate the 1-particle basis
   evaluate!(AA, tmp.tmp_basis1p, Rs, Zs, z0)

   # Stage 1: accumulate the first basis functions
   Es = zero(eltype(V))
   i1 = length(V, z0)
   for i = 1:i1
      Es = muladd(c[i], AA[i], Es)
   end

   # Stage 2: now go through the tree up to length(AA)
   #  TODO: this length should maybe be stored!
   i2 = length(AA)
   tree = V.trees[iz0]
   for i = i1+1:i2
      t = tree[i]
      a = AA[t[1]] * AA[t[2]]
      AA[i] = a
      Es = muladd(c[i], a, Es)
   end

   # Stage 3:
   i3 = length(V, z0)
   for i = i2+1:i3
      t = tree[i]
      a = AA[t[1]] * AA[t[2]]
      Es = muladd(c[i], a, Es)
   end
end




end
