
module Tree

include("extimports.jl")
include("shipimports.jl")

using Combinatorics: combinations, partitions

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



_score_partition(p) = any(isnothing, p) ? Inf : 1000 * length(p) + maximum(p)

_get_ns(p, specnew) =
      [ findfirst(isequal(kk_), specnew)  for kk_ in p ]

function _find_partition(kk, specnew)
   # @show kk
   worstp = _get_ns([ [k] for k in kk ], specnew)
   @assert worstp == kk
   bestp = worstp
   bestscore = _score_partition(bestp)

   for ip in partitions(1:length(kk))
      p = _get_ns([ kk[i] for i in ip ], specnew)
      score = _score_partition(p)
      if !any(isnothing.(p)) && score < bestscore
         bestp = p
         bestscore = score
      end
   end

   return bestp
end

# return value is the number of fake nodes added to the tree
function _insert_partition!(nodes, coeffsnew, specnew,
                            kk, p,
                            ikk, coeffsN, specN,
                            TI = Int)
   if length(p) == 2
      push!(nodes, ETNode{TI}(p[1], p[2]))
      push!(coeffsnew, coeffsN[ikk])
      push!(specnew, kk)
      return 0
   else
      # reduce the partition by pushing a new node
      push!(nodes, ETNode{TI}(p[1], p[2]))
      push!(coeffsnew, 0)
      push!(specnew, sort(vcat(specnew[p[1]], specnew[p[2]])))
      # and now recurse with the reduced partition
      return 1 + _insert_partition!(nodes, coeffsnew, specnew,
                         kk, vcat( [length(nodes)], p[3:end] ),
                         ikk, coeffsN, specN, TI)
   end
end

function get_eval_tree(inner::InnerPIBasis, coeffs;
                       filter = _->true,
                       TI = Int)
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
   coeffsnew = Vector{eltype(coeffs)}()
   sizehint!(coeffsnew, length(inner))
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
         push!(coeffsnew, 0)
      else
         push!(coeffsnew, coeffs1[ispec])
      end
   end

   # now we can construct the rest
   extranodes = 0
   for (ikk, kk) in enumerate(specN)
      # find a good partition of kk
      p = _find_partition(kk, specnew)
      extranodes += _insert_partition!(nodes, coeffsnew, specnew,
                                       kk, p, ikk, coeffsN, specN, TI)
   end

   # TODO: re-organise to minimise numstore
   numstore = maximum(maximum.(nodes))

   @info("Extra nodes inserted into the tree: $extranodes")

   return EvalTree(nodes, coeffsnew, TI(num1), TI(numstore))
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

   # Stage 2:
   # go through the tree and store the intermediate results we need
   for i = (tree.num1+1):tree.numstore
      t = nodes[i]
      a = AA[t[1]] * AA[t[2]]
      AA[i] = a
      Es = real(muladd(c[i], a, Es))
   end

   # Stage 3:
   # continue going through the tree, but now we don't need to store
   # the new correlations since the later expressions don't depend
   # on them
   for i = (tree.numstore+1):length(tree)
      t = nodes[i]
      a = AA[t[1]] * AA[t[2]]
      Es = real(muladd(c[i], a, Es))
   end

   return Es
end




end
