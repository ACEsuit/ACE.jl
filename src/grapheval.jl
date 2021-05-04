
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------



module DAG

import Base: ==, length

using Combinatorics: combinations, partitions

const BinDagNode{TI} = Tuple{TI,TI}


struct CorrEvalGraph{T, TI}
   nodes::Vector{BinDagNode{TI}}
   vals::Vector{T}
   num1::TI
   numstore::TI
end

length(dag::CorrEvalGraph) = length(dag.nodes)

==(dag1::CorrEvalGraph, dag2::CorrEvalGraph) = ACE._allfieldsequal(dag1, dag2)

CorrEvalGraph{T, TI}() where {T, TI} =
      CorrEvalGraph(Vector{BinDagNode{TI}}(undef, 0),
                    Vector{T}(undef, 0),
                    0, 0)

# -------------- FIO

# TODO: maybe there is a cleverer way to do this, for now this is just
#       a quick hack to make sure it can be read without any ambiguity

write_dict(gr::CorrEvalGraph{T, TI}) where {T <: Real, TI <: Integer} =
   Dict( "__id__" => "ACE_CorrEvalGraph",
         "T" => write_dict(T),
         "TI" => write_dict(TI),
         "nodes1" => [ n[1] for n in gr.nodes ],
         "nodes2" => [ n[2] for n in gr.nodes ],
         "vals" => gr.vals,
         "num1" => gr.num1,
         "numstore" => gr.numstore
      )


read_dict(::Val{:SHIPs_CorrEvalGraph}, D::Dict) =
   read_dict(Val{:ACE_CorrEvalGraph}(), D)

function read_dict(::Val{:ACE_CorrEvalGraph}, D::Dict)
   T = read_dict(D["T"])
   TI = read_dict(D["TI"])
   @assert T <: Real
   @assert TI <: Integer
   return CorrEvalGraph{T, TI}(
      collect(zip(D["nodes1"], D["nodes2"])),
      D["vals"],
      D["num1"],
      D["numstore"]
   )
end

# ---------------------------------------------------------------------
#   partition generator


_score_partition(p) = isempty(p) ? Inf : (1e9 * length(p) + maximum(p))

function _get_ns(p, specnew, specnew_dict)
   out = Vector{Int}(undef, length(p))
   for (i, kk_) in enumerate(p)
      if haskey(specnew_dict, kk_)
         out[i] = specnew_dict[kk_]
      else
         return Int[]
      end
   end
   return out
end


function _find_partition(kk, specnew, specnew_dict)
   worstp = _get_ns([ [k] for k in kk ], specnew, specnew_dict)
   @assert worstp == kk
   bestp = worstp
   bestscore = _score_partition(bestp)

   for ip in partitions(1:length(kk))
      p = _get_ns([ kk[i] for i in ip ], specnew, specnew_dict)
      # p_old = _get_ns_old([ kk[i] for i in ip ], specnew, specnew_dict)
      # if p != p_old
      #    @infiltrate
      # end
      score = _score_partition(p)
      if !isempty(p) && score < bestscore
         bestp = p
         bestscore = score
      end
   end

   return bestp
end


# function _insert_node!(nodes, coeffsnew, specnew, specnew_dict,
#                        newnode)



# return value is the number of fake nodes added to the dag
function _insert_partition!(nodes, coeffsnew, specnew, specnew_dict,
                            kk, p,
                            ikk, coeffsN, specN,
                            TI = Int)
   if length(p) == 2
      newnode = BinDagNode{TI}((p[1], p[2]))
      # _insert_node!(nodes, coeffsnew, specnew, specnew_dict, newnode)
      push!(nodes, newnode)
      push!(coeffsnew, coeffsN[ikk])
      push!(specnew, kk)
      specnew_dict[kk] = length(specnew)
      return 0
   else
      # @show kk, p
      # @infiltrate
      # reduce the partition by pushing a new node
      push!(nodes, BinDagNode{TI}((p[1], p[2])))
      push!(coeffsnew, 0)
      kk1 = sort(vcat(specnew[p[1]], specnew[p[2]]))
      push!(specnew, kk1)
      specnew_dict[kk1] = length(specnew)
      # and now recurse with the reduced partition
      return 1 + _insert_partition!(nodes, coeffsnew, specnew, specnew_dict,
                         kk, vcat( [length(nodes)], p[3:end] ),
                         ikk, coeffsN, specN, TI)
   end
end

function get_eval_graph(inner, coeffs;   # inner::InnerPIBasis
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
   specnew_dict = Dict{Vector{Int}, Int}()
   sizehint!(specnew, length(inner))

   # add the full 1-particle basis (N=1) into the dag
   num1 = maximum(inner.iAA2iA)
   for i = 1:num1
      push!(nodes, BinDagNode{TI}((i, 0)))
      push!(specnew, [i])
      specnew_dict[ [i] ] = length(specnew)
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
      p = _find_partition(kk, specnew, specnew_dict)
      extranodes += _insert_partition!(nodes, coeffsnew, specnew, specnew_dict,
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


function traverse_fwd!(AA, dag::CorrEvalGraph, A, fun)
   nodes = dag.nodes
   vals = dag.vals
   @assert length(AA) >= dag.numstore
   @assert length(A) >= dag.num1

   # Stage-1: copy the 1-particle basis into AA
   @inbounds for i = 1:dag.num1
      AA[i] = a = A[i]
      fun(vals[i], a)
   end

   # Stage-2: go through the dag and store the intermediate results we need
   @inbounds for i = (dag.num1+1):dag.numstore
      n1, n2 = nodes[i]
      AA[i] = a = AA[n1] * AA[n2]
      fun(vals[i], a)
   end

   # Stage 3:
   # continue going through the dag, but now we don't need to store the new
   # correlations since the later expressions don't depend on them
   @inbounds for i = (dag.numstore+1):length(dag)
      n1, n2 = nodes[i]
      a = AA[n1] * AA[n2]
      fun(vals[i], a)
   end

   return nothing
end


# function traverse_fwdbwd!(AA, dag::CorrEvalGraph, A, fwdfun, bwdfun)
#    # function fwdbwd_traverse!(dEs, tmpd, V::GraphPIPot{T}, ) where {T}
#    nodes = dag.nodes
#    vals = dag.vals
#
#    # FORWARD PASS
#    # --------------
#
#    for i = 1:dag.num1
#       AA[i] = a = A[i]
#       fwdfun(vals[i], a)
#    end
#
#    # go through the dag and store the intermediate results we need
#    for i = (dag.num1+1):dag.numstore
#       n1, n2 = nodes[i]
#       AA[i] = a = muladd(AA[n1], AA[n2], AA[i])
#       fwdfun(vals[i], a)
#    end
#
#    # go through the dag and store the intermediate results we need
#    for i = (dag.num1+1):dag.numstore
#       n1, n2 = nodes[i]
#       AA[i] = a = muladd(AA[n1], AA[n2], AA[i])
#       fwdfun(vals[i], a)
#    end
#
#
#    # BACKWARD PASS
#    # --------------
#    # fill the B array -> coefficients of the derivatives
#    #  AA_i = AA_{n1} * AA_{n2}
#    #  ∂AA_i = AA_{n1} * ∂AA_{n2} + AA_{n1} * AA_{n2}
#    #  c_{n1} * ∂AA_{n1} <- (c_{n1} + c_i AA_{n2}) ∂AA_{n1}
#    @inbounds @fastmath for i = length(dag):-1:(dag.numstore+1)
#       c = coeffs[i]
#       n1, n2 = nodes[i]
#       B[n1] = muladd(c, AA[n2], B[n1])
#       B[n2] = muladd(c, AA[n1], B[n2])
#    end
#    # in stage 2 c = C[i] is replaced with b = B[i]
#    @inbounds @fastmath for i = dag.numstore:-1:(dag.num1+1)
#       n1, n2 = nodes[i]
#       b = B[i]
#       B[n1] = muladd(b, AA[n2], B[n1])
#       B[n2] = muladd(b, AA[n1], B[n2])
#    end
#
#    # stage 3: get the gradients
#    fill!(dEs, zero(JVec{T}))
#    A = tmpd.A
#    dA = tmpd.dA
#    for (iR, (R, Z)) in enumerate(zip(Rs, Zs))
#       evaluate_d!(A, dA, tmpd_basis1p, basis1p, R, Z, z0)
#       iz = z2i(basis1p, Z)
#       inds = basis1p.Aindices[iz, iz0]
#       for iA = 1:length(basis1p, iz, iz0)
#          dEs[iR] += real(B[inds[iA]] * dA[inds[iA]])
#       end
#    end
#
#    return dEs
# end




end
