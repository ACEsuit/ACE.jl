
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


module DAG

include("extimports.jl")
include("shipimports.jl")

using Combinatorics: combinations, partitions

import SHIPs: InnerPIBasis

const BinDagNode{TI} = Tuple{TI,TI}

# struct BinDagNode{TI}
#    i1::TI
#    i2::TI
# end

struct CorrEvalGraph{T, TI}
   nodes::Vector{BinDagNode{TI}}
   vals::Vector{T}
   num1::TI
   numstore::TI
end


struct GraphPIPot{T, TI, NZ, TB} <: SitePotential
   basis1p::TB
   dags::NTuple{NZ, CorrEvalGraph{T, TI}}
end

i2z(V::GraphPIPot, i::Integer) = i2z(V.basis1p, i)
z2i(V::GraphPIPot, z::AtomicNumber) = z2i(V.basis1p, z)
numz(V::GraphPIPot) = numz(V.basis1p)

cutoff(V::GraphPIPot) = cutoff(V.basis1p)

Base.eltype(V::GraphPIPot{T}) where {T} = real(T)

_maxstore(V::GraphPIPot) = maximum( dag.numstore for dag in V.dags )

Base.length(dag::CorrEvalGraph) = length(dag.nodes)

# ---------------------------------------------------------------------
#   construction codes


function GraphPIPot(pipot::PIPotential; kwargs...)
   dags = [ get_eval_graph(pipot.pibasis.inner[iz], pipot.coeffs[iz];
                           kwargs...)  for iz = 1:numz(pipot) ]
   return GraphPIPot(pipot.pibasis.basis1p, tuple(dags...))
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

# return value is the number of fake nodes added to the dag
function _insert_partition!(nodes, coeffsnew, specnew,
                            kk, p,
                            ikk, coeffsN, specN,
                            TI = Int)
   if length(p) == 2
      push!(nodes, BinDagNode{TI}((p[1], p[2])))
      push!(coeffsnew, coeffsN[ikk])
      push!(specnew, kk)
      return 0
   else
      # reduce the partition by pushing a new node
      push!(nodes, BinDagNode{TI}((p[1], p[2])))
      push!(coeffsnew, 0)
      push!(specnew, sort(vcat(specnew[p[1]], specnew[p[2]])))
      # and now recurse with the reduced partition
      return 1 + _insert_partition!(nodes, coeffsnew, specnew,
                         kk, vcat( [length(nodes)], p[3:end] ),
                         ikk, coeffsN, specN, TI)
   end
end

function get_eval_graph(inner::InnerPIBasis, coeffs;
                       filter = _->true,
                       TI = Int,
                       verbose = false)
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

   # start assembling the dag
   nodes = BinDagNode{TI}[]
   sizehint!(nodes, length(inner))
   coeffsnew = Vector{eltype(coeffs)}()
   sizehint!(coeffsnew, length(inner))
   specnew = Vector{Int}[]
   sizehint!(specnew, length(inner))

   # add the full 1-particle basis (N=1) into the dag
   num1 = maximum(inner.iAA2iA)
   for i = 1:num1
      push!(nodes, BinDagNode{TI}((i, 0)))
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

   verbose && @info("Extra nodes inserted into the dag: $extranodes")
   numstore = length(nodes)
   num1old = num1

   # re-organise the dag layout to minimise numstore
   nodesfinal, coeffsfinal, num1, numstore = _reorder_dag!(nodes, coeffsnew)

   return CorrEvalGraph(nodesfinal, coeffsfinal, TI(num1), TI(numstore))
end


function _reorder_dag!(nodes::Vector{BinDagNode{TI}}, coeffs::Vector{T}) where {TI, T}
   # collect all AA indices that are used anywhere in the dag
   newinds = zeros(Int, length(nodes))
   newnodes = BinDagNode{TI}[]
   newcoeffs = T[]

   # inds2 = stage-2 indices, i.e. temporary storage
   # inds3 = stage-3 indices, i.e. no intermediate storage
   inds2 = sort(unique([[ n[1] for n in nodes ]; [n[2] for n in nodes]]))

   # first add all 1p nodes
   for i = 1:length(nodes)
      n, c = nodes[i], coeffs[i]
      if (n[2] == 0) ## && ((c != 0) || (i in inds2))
         @assert n[1] == i
         newinds[i] = i
         push!(newnodes, n)
         push!(newcoeffs, c)
      end
   end
   num1 = length(newnodes)

   # next add the remaining dependent nodes
   for i = 1:length(nodes)
      n, c = nodes[i], coeffs[i]
      # not 1p basis && dependent node
      if (n[2] != 0) && (i in inds2)
         push!(newnodes, BinDagNode{TI}((newinds[n[1]], newinds[n[2]])))
         push!(newcoeffs, c)
         newinds[i] = length(newnodes)
      end
   end
   numstore = length(newnodes)

   # now go through one more time and add the independent nodes
   for i = 1:length(nodes)
      n, c = nodes[i], coeffs[i]
      if (n[2] != 0) && (newinds[i] == 0) && (c != 0)
         push!(newnodes, BinDagNode{TI}((newinds[n[1]], newinds[n[2]])))
         push!(newcoeffs, c)
         newinds[i] = length(newnodes)
      end
   end

   return newnodes, newcoeffs, num1, numstore
end

# ---------------------------------------------------------------------
#   evaluation codes

alloc_temp(V::GraphPIPot{T}, maxN::Integer) where {T} =
   (
   R = zeros(JVec{real(T)}, maxN),
   Z = zeros(AtomicNumber, maxN),
   tmp_basis1p = alloc_temp(V.basis1p),
   AA = zeros(eltype(V.basis1p), _maxstore(V))
    )


function evaluate!(tmp, V::GraphPIPot, Rs, Zs, z0)
   iz0 = z2i(V, z0)
   AA = tmp.AA
   dag = V.dags[iz0]
   nodes = dag.nodes
   c = dag.vals

   # evaluate the 1-particle basis
   # this puts the first `dag.num1` 1-b (trivial) correlations into the
   # storage array, and from these we can build the rest
   evaluate!(AA, tmp.tmp_basis1p, V.basis1p, Rs, Zs, z0)

   # Stage 1: accumulate the first basis functions
   Es = zero(eltype(V))
   @inbounds for i = 1:dag.num1
      Es = muladd(c[i], real(AA[i]), Es)
   end

   # Stage 2:
   # go through the dag and store the intermediate results we need
   @inbounds @fastmath for i = (dag.num1+1):dag.numstore
      n1, n2 = nodes[i]
      AA[i] = a = AA[n1] * AA[n2]
      Es = muladd(c[i], real(a), Es)
   end

   # Stage 3:
   # continue going through the dag, but now we don't need to store
   # the new correlations since the later expressions don't depend
   # on them
   @inbounds @fastmath for i = (dag.numstore+1):length(dag)
      t = nodes[i]
      a = AA[t[1]] * AA[t[2]]
      Es = muladd(c[i], real(a), Es)
   end

   return Es
end


alloc_temp_d(V::GraphPIPot{T}, maxN::Integer) where {T} =
   (
   R = zeros(JVec{real(T)}, maxN),
   Z = zeros(AtomicNumber, maxN),
    dV = zeros(JVec{real(T)}, maxN),
   tmpd_basis1p = alloc_temp_d(V.basis1p),
   AA = zeros(eltype(V.basis1p), _maxstore(V)),
   B = zeros(eltype(V.basis1p), _maxstore(V)),
   A = alloc_B(V.basis1p),
   dA = alloc_dB(V.basis1p)
    )


function evaluate_d!(dEs, tmpd, V::GraphPIPot{T}, Rs, Zs, z0) where {T}
   iz0 = z2i(V, z0)
   AA = tmpd.AA
   B = tmpd.B
   tmpd_basis1p = tmpd.tmpd_basis1p
   basis1p = V.basis1p
   dag = V.dags[iz0]
   nodes = dag.nodes
   coeffs = dag.vals

   # we start from the representation
   #     V = sum B[i] AA[i]
   # i.e. this vector represents the constributions c[i] ∂AA[i]
   copy!(B, coeffs)

   # FORWARD PASS
   # ------------
   # evaluate the 1-particle basis
   # this puts the first `dag.num1` 1-b (trivial) correlations into the
   # storage array, and from these we can build the rest
   evaluate!(AA, tmpd_basis1p, basis1p, Rs, Zs, z0)

   # Stage 2 of evaluate!
   # go through the dag and store the intermediate results we need
   @inbounds @fastmath for i = (dag.num1+1):dag.numstore
      n1, n2 = nodes[i]
      AA[i] = muladd(AA[n1], AA[n2], AA[i])
   end

   # BACKWARD PASS
   # --------------
   # fill the B array -> coefficients of the derivatives
   #  AA_i = AA_{n1} * AA_{n2}
   #  ∂AA_i = AA_{n1} * ∂AA_{n2} + AA_{n1} * AA_{n2}
   #  c_{n1} * ∂AA_{n1} <- (c_{n1} + c_i AA_{n2}) ∂AA_{n1}
   @inbounds @fastmath for i = length(dag):-1:(dag.numstore+1)
      c = coeffs[i]
      n1, n2 = nodes[i]
      B[n1] = muladd(c, AA[n2], B[n1])
      B[n2] = muladd(c, AA[n1], B[n2])
   end
   # in stage 2 c = C[i] is replaced with b = B[i]
   @inbounds @fastmath for i = dag.numstore:-1:(dag.num1+1)
      n1, n2 = nodes[i]
      b = B[i]
      B[n1] = muladd(b, AA[n2], B[n1])
      B[n2] = muladd(b, AA[n1], B[n2])
   end

   # stage 3: get the gradients
   fill!(dEs, zero(JVec{T}))
   A = tmpd.A
   dA = tmpd.dA
   for (iR, (R, Z)) in enumerate(zip(Rs, Zs))
      evaluate_d!(A, dA, tmpd_basis1p, basis1p, R, Z, z0)
      iz = z2i(basis1p, Z)
      inds = basis1p.Aindices[iz, iz0]
      for iA = 1:length(basis1p, iz, iz0)
         dEs[iR] += real(B[inds[iA]] * dA[inds[iA]])
      end
   end

   return dEs
end



end
