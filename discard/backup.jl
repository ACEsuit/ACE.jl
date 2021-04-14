

"""
`@generated function nfcalls(::Val{N}, f)`

Effectively generates a loop of functions calls, but fully unrolled and
therefore type-stable:
```{julia}
f(Val(1))
f(Val(2))
f(Val(3))
# ...
f(Val(N))
```
"""
@generated function nfcalls(::Val{N}, f) where {N}
   code = Expr[]
   for n = 1:N
      push!(code, :(f(Val($n))))
   end
   quote
      $(Expr(:block, code...))
      return nothing
   end
end


"""
`@generated function valnmapreduce(::Val{N}, v, f)`

Generates a map-reduce like code, with fully unrolled loop which makes this
type-stable,
```{julia}
begin
   v += f(Val(1))
   v += f(Val(2))
   # ...
   v += f(Val(N))
   return v
end
```
"""
@generated function valnmapreduce(::Val{N}, v, f) where {N}
   code = Expr[]
   for n = 1:N
      push!(code, :(v += f(Val($n))))
   end
   quote
      $(Expr(:block, code...))
      return v
   end
end



# ----------------------------------------------------------------------
# sparse matrix multiplication with weaker type restrictions


function _my_mul!(C::AbstractVector, A::SparseMatrixCSC, B::AbstractVector)
   A.n == length(B) || throw(DimensionMismatch())
   A.m == length(C) || throw(DimensionMismatch())
   nzv = A.nzval
   rv = A.rowval
   fill!(C, zero(eltype(C)))
   @inbounds for col = 1:A.n
      b = B[col]
      for j = A.colptr[col]:(A.colptr[col + 1] - 1)
         C[rv[j]] += nzv[j] * b
      end
   end
   return C
end



# # gradients
#
# site_alloc_dB(basis::PIBasis, Rs::AbstractVector, args...) =
#    site_alloc_dB(basis, length(Rs))
#
# site_alloc_dB(basis::PIBasis, nmax::Integer) =
#       zeros( JVec{fltype(basis)}, maximum(length.(basis.inner)), nmax )
#
# alloc_temp_d(basis::PIBasis, Rs::AbstractVector, args...) =
#    alloc_temp_d(basis, length(Rs))
#
# alloc_temp_d(basis::PIBasis, nmax::Integer) =
#       (
#         A = alloc_B(basis.basis1p, nmax),
#         dA = alloc_dB(basis.basis1p, nmax),
#         tmp_basis1p = alloc_temp(basis.basis1p, nmax),
#         tmpd_basis1p = alloc_temp_d(basis.basis1p, nmax),
#         tmpd_inner = alloc_temp_d(basis.evaluator, basis, nmax)
#       )
#
#
# function evaluate_d!(AA, dAA, tmpd, basis::PIBasis, Rs, Zs, z0)
#    iz0 = z2i(basis, z0)
#    AAview = @view AA[basis.inner[iz0].AAindices]
#    dAAview = @view dAA[basis.inner[iz0].AAindices, 1:length(Rs)]
#    site_evaluate_d!(AAview, dAAview, tmpd, basis, Rs, Zs, z0)
#    return dAA
# end
#
# function site_evaluate_d!(AA, dAA, tmpd, basis::PIBasis, Rs, Zs, z0)
#    iz0 = z2i(basis, z0)
#    # precompute the 1-p basis and its derivatives
#    evaluate_d!(tmpd.A, tmpd.dA, tmpd.tmpd_basis1p, basis.basis1p, Rs, Zs, z0)
#    site_evaluate_d!(AA, dAA, tmpd, basis.inner[iz0], basis.evaluator, tmpd.A, tmpd.dA)
#    return nothing
# end
#
#
# #--------------------------------------------------------
# # ------- specialised code for the standard evaluator
#
# alloc_temp(::StandardEvaluator, basis::PIBasis) = nothing
#
# alloc_temp_d(::StandardEvaluator, basis::PIBasis, args...) = nothing
#
# function site_evaluate_d!(AA, dAA, tmpd, inner::InnerPIBasis,
#                           _ev::StandardEvaluator, A, dA)
#    # evaluate the AA basis
#    evaluate!(AA, nothing, inner, _ev, tmpd.A)
#    # loop over all neighbours
#    for j = 1:size(dA, 2)
#       # write the gradients into the correct slice of the dAA matrix
#       for iAA = 1:length(inner)
#          @inbounds dAA[iAA, j] = grad_AAi_Rj(iAA, j, inner, A, dA)
#       end
#    end
#    return nothing
# end
#
#
#
# """
# Compute ∂∏A_a / ∂A_b = ∏_{a ≂̸ b} A_a
# """
# function grad_AAi_Ab(iAA, b, inner, A)
#    g = one(eltype(A))
#    @inbounds for a = 1:inner.orders[iAA]
#       if a != b
#          g *= A[inner.iAA2iA[iAA, a]]
#       end
#    end
#    return g
# end
#
# @doc raw"""
# Compuate ∂∏A_a / ∂Rⱼ:
# ```math
#    \frac{\partial \prod_a A_a}{\partial \bm r_j}
#    =
#    \sum_b \frac{\prod_{a} A_a}{\partial A_b} \cdot \frac{\partial A_b}{\partial \bm r_j}
#    =
#    \sum_b \prod_{a \neq b} A_a \cdot \frac{\partial \phi_b}{\partial \bm r_j}
# ```
# """
# function grad_AAi_Rj(iAA, j, inner, A, dA)
#    g = zero(eltype(dA))
#    @inbounds for b = 1:inner.orders[iAA] # interaction order
#       # A_{k_b} = A[iA]
#       iA = inner.iAA2iA[iAA, b]
#       # dAAi_dAb = ∂(∏A_{n_a}) / ∂A_{n_b}
#       dAAi_dAb = grad_AAi_Ab(iAA, b, inner, A)
#       g += dAAi_dAb * dA[iA, j]
#    end
#    return g
# end
#
#
# # --------------------------------------------------------
# #   Evaluation code for DAG evaluator
#
#
# alloc_temp(::DAGEvaluator, basis::PIBasis) =
#    zeros(fltype(basis), maximum(inner.dag.numstore for inner in basis.inner))
#
#
# function alloc_temp_d(::DAGEvaluator, basis::PIBasis,  nmax::Integer)
#    maxstore = maximum(inner.dag.numstore for inner in basis.inner)
#    return (
#       AA = zeros(fltype(basis), maxstore),
#       dAA = zeros(JVec{fltype(basis)}, maxstore, nmax)
#    )
# end
#
# function evaluate!(AA, tmp, basis::InnerPIBasis, ::DAGEvaluator, A)
#    AAdag = tmp.tmp_inner
#    fill!(AAdag, 1)
#    traverse_fwd!(AAdag, basis.dag, A,
#                  (idx, AAval) -> if idx > 0; AA[idx] = AAval; end)
#    return AA
# end
#
#
#
# function site_evaluate_d!(AA, dAA, tmpd, inner::InnerPIBasis,
#                           ::DAGEvaluator, A, dA)
#    dag = inner.dag
#    nodes = dag.nodes
#    idxs = dag.vals
#    AAtmp = tmpd.tmpd_inner.AA
#    dAAtmp = tmpd.tmpd_inner.dAA
#    numneigs = size(dA, 2)
#    # --- manual bounds checking
#    @assert size(dAA, 2) >= numneigs
#    @assert size(dAAtmp, 2) >= numneigs
#    @assert length(AAtmp) >= inner.dag.numstore
#    @assert size(dAAtmp, 1) >= inner.dag.numstore
#    # ------------------
#
#    @inline function _copyrow!(A_, iA, B_, iB)
#       @simd for j = 1:numneigs
#          @inbounds A_[iA,j] = B_[iB,j]
#       end
#    end
#
#    @inline function _fwdgrad!(dAA_, i, n1, AA1, n2, AA2)
#       @simd for j = 1:numneigs
#          @inbounds dAA_[i, j] = AA1 * dAAtmp[n2,j] + AA2 * dAAtmp[n1,j]
#       end
#    end
#
#    @inbounds for i = 1:dag.num1
#       idx = idxs[i]
#       AAtmp[i] = A[i]
#       _copyrow!(dAAtmp, i, dA, i)
#       if idx > 0
#          AA[idx] = AAtmp[i]
#          _copyrow!(dAA, idx, dAAtmp, i)
#          # @. dAA[idx,:] = (@view dAAtmp[i,:])
#       end
#    end
#
#    @inbounds for i = (dag.num1+1):dag.numstore
#       n1, n2 = nodes[i]
#       idx = idxs[i]
#       AA1, AA2 = AAtmp[n1], AAtmp[n2]
#       AAtmp[i] = AA1 * AA2
#       _fwdgrad!(dAAtmp, i, n1, AA1, n2, AA2)
#       # @. dAAtmp[i, :] = AA1 * (@view dAAtmp[n2,:]) + AA2 * (@view dAAtmp[n1,:])
#       if idx > 0
#          AA[idx] = AAtmp[i]
#          _copyrow!(dAA, idx, dAAtmp, i)
#          # @. dAA[idx, :] .= (@view dAAtmp[i, :])
#       end
#    end
#
#    @inbounds for i = (dag.numstore+1):length(dag)
#       idx = idxs[i]
#       n1, n2 = nodes[i]
#       # @assert idx > 0
#       AA1, AA2 = AAtmp[n1], AAtmp[n2]
#       AA[idx] = AA1 * AA2
#       _fwdgrad!(dAA, idx, n1, AA1, n2, AA2)
#       # @simd for j = 1:numneigs
#       #    @inbounds dAA[idx, j] = AA1 * dAAtmp[n2,j] + AA2 * dAAtmp[n1,j]
#       # end
#    end
#
#    return nothing
# end



   # ----- compute the adjoints dAA / dA
   for iAA = 1:length(basis)
      dAAdA[1] = 1
      AAfwd = A[iAA2iA[iAA, 1]]
      ord = orders[iAA]
      for a = 2:ord
         dAAdA[a] = AAfwd
         AAfwd *= A[iAA2iA[iAA, a]]
      end
      AA[iAA] = AAfwd
      AAbwd = A[iAA2iA[iAA, ord]]
      for a = ord-1:-1:1
         dAAdA[a] *= AAbwd
         AAbwd *= A[iAA2iA[iAA, a]]
      end
      for a = 1:ord
         W[iAA2iA[iAA, a]] += dAAdA[a]
      end
   end
