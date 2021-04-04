
export PIBasis


# ---------------------- Implementation of the PIBasisSpec

"""
`struct PIBasisSpec`
"""
struct PIBasisSpec
   orders::Vector{Int}     # order (length) of ith basis function
   iAA2iA::Matrix{Int}     # where in A can we find the ith basis function
end

==(B1::PIBasisSpec, B2::PIBasisSpec) = (
         (B1.b2iA == B2.b2iA) &&
         (B1.iAA2iA == B2.iAA2iA) )

Base.length(spec::PIBasisSpec) = length(spec.orders)


function _get_pibfcn(spec0, Aspec, vv)
   vv1 = vv[2:end]
   vvnz = vv1[findall(vv1 .!= 0)]
   return (spec0[vv[1]], Aspec[vvnz])
end

function _get_pibfcn(Aspec, vv)
   vvnz = vv[findall(vv .!= 0)]
   return Aspec[vvnz]
end


function PIBasisSpec( basis1p::OneParticleBasis,
                      maxν::Integer, maxdeg::Real;
                      Deg = NaiveTotalDegree(),
                      property = nothing,
                      filterfun = _->true,
                      constant = false )
   # would make sense to construct the basis1p spec here?

   # get the basis spec of the one-particle basis
   #  Aspec[i] described the basis function that will get written into A[i]
   Aspec = get_spec(basis1p)

   # we assume that `Aspec` is sorted by degree, but best to double-check this
   # since the notion of degree used to construct `Aspec` might be different
   # from the one used to construct AAspec.
   if !issorted(Aspec; by = b -> degree(b, Deg, basis1p))
      error("""PIBasisSpec : AAspec construction failed because Aspec is not
               sorted by degree. This could e.g. happen if an incompatible
               notion of degree was used to construct the 1-p basis spec.""")
   end
   # An AA basis function is given by a tuple 𝒗 = vv. Each index 𝒗ᵢ = vv[i]
   # corresponds to the basis function Aspec[𝒗ᵢ] and the tuple
   # 𝒗 = (𝒗₁, ...) to a product basis function
   #   ∏ A_{vₐ}
   tup2b = vv -> _get_pibfcn(Aspec, vv)

   #  degree of a basis function ↦ is it admissible?
   admissible = b -> (degree(b, Deg, basis1p) <= maxdeg)

   if property != nothing
      filter1 = b -> filterfun(b) && filter(property, b)
   else
      filter1 = filterfun
   end


   # we can now construct the basis specification; the `ordered = true`
   # keyword signifies that this is a permutation-invariant basis
   AAspec = gensparse(; NU = maxν,
                        tup2b = tup2b,
                        admissible = admissible,
                        ordered = true,
                        maxvv = [length(Aspec) for _=1:maxν],
                        filter = filter1,
                        constant = constant )

   return PIBasisSpec(AAspec)
end


function PIBasisSpec(AAspec)
   orders = zeros(Int, length(AAspec))
   iAA2iA = zeros(Int, (length(AAspec), length(AAspec[1])))
   for (iAA, vv) in enumerate(AAspec)
      # we use reverse because gensparse constructs the indices in
      # ascending order, but we want descending here.
      # (I don't remember why though)
      iAA2iA[iAA, :] .= reverse(vv)
      orders[iAA] = length( findall( vv .!= 0 ) )
   end
   return PIBasisSpec(orders, iAA2iA)
end


get_spec(AAspec::PIBasisSpec, i::Integer) = AAspec.iAA2iA[i, 1:AAspec.orders[i]]


# --------------------------------- PIBasis implementation


"""
`mutable struct PIBasis:` implementation of a permutation-invariant
basis based on the density projection trick.

The standard constructor is
```
PIBasis(basis1p, N, D, maxdeg)
```
* `basis1p` : a one-particle basis
* `N` : maximum interaction order
* `D` : an abstract degee specification, e.g., SparsePSHDegree
* `maxdeg` : the maximum polynomial degree as measured by `D`
"""
mutable struct PIBasis{BOP, REAL} <: ACEBasis
   basis1p::BOP             # a one-particle basis
   spec::PIBasisSpec
   real::REAL     # could be `real` or `identity` to keep AA complex
   # evaluator    # classic vs graph
end

cutoff(basis::PIBasis) = cutoff(basis.basis1p)

==(B1::PIBasis, B2::PIBasis) = ACE._allfieldsequal(B1, B2)

# TODO: allow the option of converting to real part?
fltype(basis::PIBasis) = basis.real( fltype(basis.basis1p) )
rfltype(basis::PIBasis) = real( fltype(basis) )

Base.length(basis::PIBasis) = length(basis.spec)

PIBasis(basis1p, args...; isreal = true, kwargs...) =
   PIBasis(basis1p, PIBasisSpec(basis1p, args...; kwargs...),
           isreal ? Base.real : Base.identity )


get_spec(pibasis::PIBasis) =
   [ get_spec(pibasis, i) for i = 1:length(pibasis) ]

get_spec(pibasis::PIBasis, i::Integer) =
      get_spec.( Ref(pibasis.basis1p), get_spec(pibasis.spec, i) )

setreal(basis::PIBasis, isreal::Bool) =
   PIBasis(basis.basis1p, basis.spec, isreal)


# function scaling(pibasis::PIBasis, p)
#    ww = zeros(Float64, length(pibasis))
#    for iz0 = 1:numz(pibasis)
#       wwin = @view ww[pibasis.inner[iz0].AAindices]
#       for i = 1:length(pibasis.inner[iz0])
#          bspec = get_basis_spec(pibasis, iz0, i)
#          wwin[i] = scaling(bspec, p)
#       end
#    end
#    return ww
# end



# graphevaluator(basis::PIBasis) =
#    PIBasis(basis.basis1p, zlist(basis), basis.inner, DAGEvaluator())
#
# standardevaluator(basis::PIBasis) =
#    PIBasis(basis.basis1p, zlist(basis), basis.inner, StandardEvaluator())
#
#
#
# -------------------------------------------------
# FIO codes

write_dict(basis::PIBasis) =
   Dict(  "__id__" => "ACE_PIBasis",
         "basis1p" => write_dict(basis.basis1p),
            "spec" => write_dict(basis.spec),
            "real" => basis.real )

read_dict(::Val{:ACE_PIBasis}, D::Dict) =
   PIBasis( read_dict(D["basis1p"]),
            read_dict(D["spec"]),
            D["real"] )

write_dict(spec::PIBasisSpec) =
   Dict( "__id__" => "ACE_PIBasisSpec",
         "orders" => spec.orders,
         "iAA2iA" => write_dict(spec.iAA2iA) )

read_dict(::Val{:ACE_PIBasisSpec}, D::Dict) =
   PIBasisSpec( D["orders"], read_dict(D["iAA2iA"]) )


# -------------------------------------------------
# Evaluation codes

alloc_B(basis::PIBasis, args...) = zeros( fltype(basis), length(basis) )

alloc_temp(basis::PIBasis, args...) =
      ( A = alloc_B(basis.basis1p, args...),
        tmp1p = alloc_temp(basis.basis1p, args...),
      )


function evaluate!(AA, tmp, basis::PIBasis, config::AbstractConfiguration)
   A = evaluate!(tmp.A, tmp.tmp1p, basis.basis1p, config)
   fill!(AA, 1)
   for iAA = 1:length(basis)
      aa = one(eltype(A))
      for t = 1:basis.spec.orders[iAA]
         aa *= A[ basis.spec.iAA2iA[ iAA, t ] ]
      end
      AA[iAA] = basis.real(aa)
   end
   return AA
end




# -------------------------------------------------


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
# # this method evaluates the InnerPIBasis, which is really the actual
# # evaluation code; the rest is just bookkeeping
# function evaluate!(AA, tmp, basis::InnerPIBasis, ::StandardEvaluator, A)
#    fill!(AA, 1)
#    for i = 1:length(basis)
#       for α = 1:basis.orders[i]
#          iA = basis.iAA2iA[i, α]
#          AA[i] *= A[iA]
#       end
#    end
#    return AA
# end
#
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
