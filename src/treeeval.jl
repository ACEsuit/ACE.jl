
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

const ETNode{TI} = Tuple{TI,TI}

ETNode{TI}(i1, i2) where {TI} = ( TI(i1), TI(i2) )


struct EvalTree{T, TI}
   nodes::Vector{ETNode{TI}}
   coeffs::Vector{T}
   num1::TI
   numstore::TI
end


struct TreePIPot{T, TI, NZ, TB} <: SitePotential
   basis1p::TB
   trees::NTuple{NZ, EvalTree{T, TI}}
end

i2z(V::TreePIPot, i::Integer) = i2z(V.basis1p, i)
z2i(V::TreePIPot, z::AtomicNumber) = z2i(V.basis1p, z)
numz(V::TreePIPot) = numz(V.basis1p)

cutoff(V::TreePIPot) = cutoff(V.basis1p)

Base.eltype(V::TreePIPot{T}) where {T} = real(T)

_maxstore(V::TreePIPot) = maximum( tree.numstore for tree in V.trees )

Base.length(tree::EvalTree) = length(tree.nodes)

# ---------------------------------------------------------------------
#   construction codes


function TreePIPot(pipot::PIPotential; kwargs...)
   trees = [ get_eval_tree(pipot.pibasis.inner[iz], pipot.coeffs[iz];
                           kwargs...)  for iz = 1:numz(pipot) ]
   return TreePIPot(pipot.pibasis.basis1p, tuple(trees...))
end




function get_eval_tree(inner::InnerPIBasis, coeffs;
                       filter = _->true,
                       TI = UInt16)
   # make a list of all basis functions as vectors so we can search it
   # TODO: should also check the tuples are sorted lexicographically
   spec = [ inner.iAA2iA[iAA, 1:inner.orders[iAA]]
            for iAA = 1:length(inner) ]
   @assert issorted(length.(spec))
   @assert all(issorted, spec)
   # we need to separate them into 1-p and many-p
   spec1 = spec[ length.(spec) .== 1 ]
   coeffs1 = coeffs[1:length(spec1)]
   IN = (length(spec1)+1):length(spec)
   specN = spec[IN]
   coeffsN = coeffs[IN]

   # start assembling the tree
   nodes = ETNode{TI}[]
   sizehint!(nodes, length(inner))
   newcoeffs = Vector{eltype(coeffs)}()
   sizehint!(newcoeffs, length(inner))
   specnew = Vector{Int}[]
   sizehint!(specnew, length(inner))

   # add the full 1-particle basis (N=1) into the tree
   num1 = maximum(inner.iAA2iA)
   for i = 1:num1
      push!(nodes, ETNode{TI}(i, 0))
      push!(specnew, [i])
      # find that index in `spec`
      ispec = findfirst(isequal([i]), spec1)
      if isnothing(ispec)
         push!(newcoeffs, 0)
      else
         push!(newcoeffs, coeffs1[ispec])
      end
   end

   # now we can construct the rest
   # this is the limit of how many intermediate computations must be stored
   for (ikk, kk) in enumerate(specN)
      if length(kk) == 1; continue; end # skip the 1-p, we already have them
      # best length
      bestlen = length(kk)
      best_node = ETNode{TI}(0,0)
      for i1 in combinations(1:length(kk))
         if length(i1) > length(kk) ÷ 2; continue; end
         i2 = setdiff(1:length(kk), i1)
         kk1 = sort(kk[i1])
         kk2 = sort(kk[i2])
         len = max( length(kk1), length(kk2) )
         if len < bestlen
            n1 = findfirst(isequal(kk1), specnew)
            n2 = findfirst(isequal(kk2), specnew)
            if n1 != nothing && n2 != nothing
               bestlen = len
               best_node = ETNode{TI}(n1, n2)
            end
         end
      end
      if best_node == ETNode{TI}(0,0)
         error("couldn't find a decomposition for $kk")
      end
      # add into the collection
      push!(nodes, best_node)
      push!(newcoeffs, coeffs[ikk])
      push!(specnew, kk)
   end
   numstore = maximum(maximum.(nodes))
   return EvalTree(nodes, newcoeffs, TI(num1), TI(numstore))
end




# ---------------------------------------------------------------------
#   evaluation codes

alloc_temp(V::TreePIPot{T}, maxN::Integer) where {T} =
   (
   R = zeros(JVec{real(T)}, maxN),
   Z = zeros(AtomicNumber, maxN),
   tmp_basis1p = alloc_temp(V.basis1p),
   A = alloc_B(V.basis1p),
   AA = zeros(eltype(V.basis1p), _maxstore(V))
    )


function evaluate!(tmp, V::TreePIPot, Rs, Zs, z0)
   iz0 = z2i(V, z0)
   AA = tmp.AA
   tree = V.trees[iz0]
   nodes = tree.nodes
   c = tree.coeffs

   # evaluate the 1-particle basis
   # this puts the first `tree.num1` 1-b (trivial) correlations into the
   # storage array, and from these we can build the rest
   evaluate!(AA, tmp.tmp_basis1p, V.basis1p, Rs, Zs, z0)

   # Stage 1: accumulate the first basis functions
   Es = zero(eltype(V))
   for i = 1:tree.num1
      Es = real(muladd(c[i], AA[i], Es))
   end

   # Stage 2: now go through the tree and store the intermediate results we need
   for i = (tree.num1+1):tree.numstore
      t = nodes[i]
      a = AA[t[1]] * AA[t[2]]
      AA[i] = a
      Es = real(muladd(c[i], a, Es))
   end

   # Stage 3:
   # continue going through the tree, but now we don't need to store
   # the new correlations since we don't need them anymore later
   for i = (tree.numstore+1):length(tree)
      t = nodes[i]
      a = AA[t[1]] * AA[t[2]]
      Es = real(muladd(c[i], a, Es))
   end

   return Es
end




end
