
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------




import ACE.DAG: CorrEvalGraph, get_eval_graph, traverse_fwd!

export PIBasis

export graphevaluator, standardevaluator


"""
`PIBasisFcn{N, TOP}` : represents a single multivariate basis function
in terms of 1-particle pasis functions in each coordinate direction. Crucially,
this function will be interpreted as a *permutation invariant* basis function!
"""
struct PIBasisFcn{N, TOP <: OnepBasisFcn}
   z0::AtomicNumber
   oneps::NTuple{N, TOP}
   top::Type{TOP}
end

_top(::PIBasisFcn{N, TOP}) where {N, TOP} = TOP

PIBasisFcn(z0::AtomicNumber, oneps) =
   PIBasisFcn(z0, tuple(oneps...), typeof(oneps[1]))

order(b::PIBasisFcn{N}) where {N} = N

degree(d::AbstractDegree, pphi::PIBasisFcn) = degree(d, pphi.oneps)

# TODO: this is very rough - can we do better?
scaling(b::PIBasisFcn, p) = sum(scaling(bb, p) for bb in b.oneps)

# TODO: can we replace this with get_basis_spec?
function PIBasisFcn(Aspec, t, z0::AtomicNumber)
   if isempty(t) || sum(abs, t) == 0
      TOP = eltype(Aspec)
      return PIBasisFcn{0, eltype(Aspec)}(z0, NTuple{0, TOP}(), TOP)
   end
   # zeros stand for reduction in body-order
   tnz = t[findall(t .!= 0)]
   return PIBasisFcn(z0, Aspec[[tnz...]])
end

write_dict(b::PIBasisFcn) =
   Dict("__id__" => "ACE_PIBasisFcn",
        "z0" => write_dict(b.z0),
        "oneps" => write_dict.(b.oneps))

read_dict(::Val{:SHIPs_PIBasisFcn}, D::Dict) =
   read_dict(Val{:ACE_PIBasisFcn}(), D::Dict)

read_dict(::Val{:ACE_PIBasisFcn}, D::Dict) =
   PIBasisFcn( read_dict(D["z0"]),
               tuple( read_dict.(D["oneps"]) ... ) )


"""
note this function doesn't return an ordered specification, this is due
to the fact that we don't require the Aspec to be ordered by degree.
Instead the ordering is achieved in the InnerPIBasis constructor
"""
function get_PI_spec(basis1p::OneParticleBasis, N::Integer,
                     degree::AbstractDegree, maxdeg::Real,
                     z0::AtomicNumber; filter = _->true )
   iz0 = z2i(basis1p, z0)
   # get the basis spec of the one-particle basis
   #  Aspec[i] described the basis function that will get written into A[i]
   #  but we don't care here since we will just map back and forth in the
   #  pre-computation stage. note AAspec below will not store indices to Aspec
   #  but the actual basis functions themselves.
   Aspec = get_basis_spec(basis1p, z0)
   # next we need to sort it by degree so that gensparse doesn't get confused.
   Aspec_p = sort(Aspec, by = b -> degree(b, z0))
   # now an index νi corresponds to the basis function
   # Aspec[p[νi]] = Aspec_p[νi] and a tuple ν = (ν1,...,νN) to the following
   # basis function
   tup2b = ν -> PIBasisFcn(Aspec_p, ν, z0)
   # we can now construct the basis specification; the `ordered = true`
   # keyword signifies that this is a permutation-invariant basis
   AAspec = gensparse(N, maxdeg;
                      tup2b = tup2b, degfun = degree, ordered = true,
                      maxν = length(Aspec_p),
                      filter = filter)
   return AAspec
end


"""
`mutable struct InnerPIBasis` : this type is just an auxilary type to
make the implementation of `PIBasis` clearer. It implements the
permutation-invariant basis for a single centre-atom species. The main
type `PIBasis` then stores `NZ` objects of type `InnerPIBasis`
and "dispatches" the work accordingly.
"""
mutable struct InnerPIBasis
   orders::Vector{Int}           # order (length) of ith basis function
   iAA2iA::Matrix{Int}           # where in A can we find the ith basis function
   b2iAA::Dict{PIBasisFcn, Int}  # mapping PIBasisFcn -> iAA =  index in AA[z0]
   b2iA::Dict{Any, Int}          # mapping from 1-p basis fcn to index in A[z0]
   AAindices::UnitRange{Int}     # where in AA does AA[z0] fit?
   z0::AtomicNumber              # inner basis for which species?
   dag::CorrEvalGraph{Int, Int}  # for fast evaluation
end


==(B1::InnerPIBasis, B2::InnerPIBasis) = (
   (B1.b2iA == B2.b2iA) &&
   (B1.iAA2iA == B2.iAA2iA) )

maxorder(basis::InnerPIBasis) = size(basis.iAA2iA, 2)

Base.length(basis::InnerPIBasis) = length(basis.orders)

function InnerPIBasis(Aspec, AAspec, AAindices, z0)
   len = length(AAspec)
   maxorder = maximum(order, AAspec)

   # construct the b2iA mapping
   b2iA = Dict{Any, Int}()
   for (iA, b) in enumerate(Aspec)
      if haskey(b2iA, b)
         @show b
         error("b2iA already has the key b. This means the basis spec is invalid.")
      end
      b2iA[b] = iA
   end

   # allocate the two main arrays used for evaluation ...
   orders = zeros(Int, len)
   iAA2iA = zeros(Int, len, maxorder)
   # ... and fill them up with the cross-indices
   for (iAA, b) in enumerate(AAspec)
      @assert b.z0 == z0
      b = _get_ordered(b2iA, b)
      AAspec[iAA] = b

      orders[iAA] = order(b)
      for α = 1:order(b)
         iAA2iA[iAA, α] = b2iA[ b.oneps[α] ]
      end
   end

   # now sort iAA2iA lexicographically by rows to make sure the representation
   perm = sortperm( [ vcat([orders[iAA]], iAA2iA[iAA,:]) for iAA = 1:len ] )
   iAA2iA = iAA2iA[perm, :]
   AAspec = AAspec[perm]
   orders = orders[perm]

   # construct the b2iAA mapping
   b2iAA = Dict{PIBasisFcn, Int}()
   for (iAA, b) in enumerate(AAspec)
      if haskey(b2iAA, b)
         @show b
         error("b2iAA already has the key b. This means the basis spec is invalid.")
      end
      b2iAA[b] = iAA
   end

   # putting it all together :
   # an empty dag, to assemble the basis ...
   emptydag = DAG.CorrEvalGraph{Int, Int}()
   inner = InnerPIBasis(orders, iAA2iA, b2iAA, b2iA, AAindices, z0, emptydag)
   # ... generate the actual dag to return
   generate_dag!(inner)
   return inner
end


function generate_dag!(inner; kwargs...)
   len = length(inner)
   inner.dag = get_eval_graph(inner, collect(1:len); kwargs...)
   return inner
end


# ---------------------- PIBasis

struct DAGEvaluator end
struct StandardEvaluator end


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

Note the species list will be taken from `basis1p`
"""
mutable struct PIBasis{BOP, NZ, TEV} <: IPBasis
   basis1p::BOP             # a one-particle basis
   zlist::SZList{NZ}
   inner::NTuple{NZ, InnerPIBasis}
   evaluator::TEV
end

cutoff(basis::PIBasis) = cutoff(basis.basis1p)

==(B1::PIBasis, B2::PIBasis) = ACE._allfieldsequal(B1, B2)

fltype(basis::PIBasis) = fltype(basis.basis1p)

Base.length(basis::PIBasis) = sum(length(basis, iz) for iz = 1:numz(basis))
Base.length(basis::PIBasis, iz0::Integer) = length(basis.inner[iz0])
Base.length(basis::PIBasis, z0::AtomicNumber) = length(basis, z2i(basis, z0))

function PIBasis(basis1p::OneParticleBasis,
                 N::Integer,
                 D::AbstractDegree,
                 maxdeg::Real;
                 filter = b -> order(b) > 0)
   innerspecs = Any[]
   # now for each iz0 (i.e. for each z0) construct an "inner basis".
   for iz0 = 1:numz(basis1p)
      z0 = i2z(basis1p, iz0)
      # get a list of 1-p basis function
      AAspec_z0 = get_PI_spec(basis1p, N, D, maxdeg, z0; filter=filter)
      push!(innerspecs, AAspec_z0)
   end
   return pibasis_from_specs(basis1p, identity.(innerspecs))
end



# TODO: instead of copying zlist, maybe forward the zlist methods

function pibasis_from_specs(basis1p, innerspecs)
   idx = 0
   inner = InnerPIBasis[]
   for iz0 = 1:numz(basis1p)
      z0 = i2z(basis1p, iz0)
      Aspec_z0 = get_basis_spec(basis1p, z0)
      AAspec_z0 = innerspecs[iz0]
      AAindices = (idx+1):(idx+length(AAspec_z0))
      push!(inner, InnerPIBasis(Aspec_z0, AAspec_z0, AAindices, z0))
      idx += length(AAspec_z0)
   end
   return PIBasis(basis1p, zlist(basis1p), tuple(inner...), DAGEvaluator())
end


"""
`get_basis_spec(basis::PIBasis, iz0::Integer, i::Integer)`

Here `i` is the index of the basis function for which we reconstruct its
specification.
"""
function get_basis_spec(basis::PIBasis, iz0::Integer, i::Integer)
   N = basis.inner[iz0].orders[i]
   iAA2iA = basis.inner[iz0].iAA2iA[i, 1:N]
   if N == 0
      # TODO - Nasty hack; I'm assuming all TOPs are the same
      TOP = typeof(get_basis_spec(basis.basis1p, iz0, 1))
      return PIBasisFcn(i2z(basis, iz0), NTuple{0, TOP}(), TOP)
   end
   return PIBasisFcn( i2z(basis, iz0),
                  [ get_basis_spec(basis.basis1p, iz0, iAA2iA[n]) for n = 1:N] )
end

function _get_ordered(pibasis::PIBasis, pib::PIBasisFcn)
   inner = pibasis.inner[z2i(pibasis, pib.z0)]
   return _get_ordered(inner.b2iA, pib)
end

_get_ordered(::Dict, pib::PIBasisFcn{0}) = pib
_get_ordered(::Dict, pib::PIBasisFcn{1}) = pib

function _get_ordered(b2iA::Dict, pib::PIBasisFcn{N}) where {N}
   iAs = [ b2iA[b] for b in pib.oneps ]
   p = sortperm(iAs)
   return PIBasisFcn(pib.z0, ntuple(i -> pib.oneps[p[i]], N))
end


function scaling(pibasis::PIBasis, p)
   ww = zeros(Float64, length(pibasis))
   for iz0 = 1:numz(pibasis)
      wwin = @view ww[pibasis.inner[iz0].AAindices]
      for i = 1:length(pibasis.inner[iz0])
         bspec = get_basis_spec(pibasis, iz0, i)
         wwin[i] = scaling(bspec, p)
      end
   end
   return ww
end


graphevaluator(basis::PIBasis) =
   PIBasis(basis.basis1p, zlist(basis), basis.inner, DAGEvaluator())

standardevaluator(basis::PIBasis) =
   PIBasis(basis.basis1p, zlist(basis), basis.inner, StandardEvaluator())

maxorder(basis::PIBasis) = maximum(maxorder.(basis.inner))

# -------------------------------------------------
# FIO codes

# TODO: at the moment the DAGs will be generated from scratch
#       every time we load the basis...

write_dict(basis::PIBasis) =
   Dict(  "__id__" => "ACE_PIBasis",
         "basis1p" => write_dict(basis.basis1p),
           "inner" => [ write_dict.( collect(keys(basis.inner[iz0].b2iAA)) )
                         for iz0 = 1:numz(basis) ], )

read_dict(::Val{:SHIPs_PIBasis}, D::Dict) =
   read_dict(Val{:ACE_PIBasis}(), D::Dict)

function read_dict(::Val{:ACE_PIBasis}, D::Dict)
   basis1p = read_dict(D["basis1p"])
   innerspecs = [ read_dict.(D["inner"][iz0])  for iz0 = 1:numz(basis1p) ]
   return pibasis_from_specs(basis1p, innerspecs)
end


# -------------------------------------------------
# Evaluation codes

site_alloc_B(basis::PIBasis, args...) =
      zeros( fltype(basis), maximum(length.(basis.inner)) )

alloc_temp(basis::PIBasis, args...) =
      ( A = alloc_B(basis.basis1p, args...),
        tmp_basis1p = alloc_temp(basis.basis1p, args...),
        tmp_inner = alloc_temp(basis.evaluator, basis)
      )


evaluate!(AA, tmp, basis::PIBasis, args...) =
   evaluate!(AA, tmp, basis::PIBasis, basis.evaluator, args...)

# this method treats basis as a actual basis across all species
function evaluate!(AA, tmp, basis::PIBasis, Rs, Zs, z0)
   fill!(AA, 0)
   iz0 = z2i(basis, z0)
   AAview = @view AA[basis.inner[iz0].AAindices]
   site_evaluate!(AAview, tmp, basis, Rs, Zs, z0)
   return AA
end

# this method treats basis as a basis for a single species z0, i.e.
# AA is assumed to only contain the AA^{z0}.
function site_evaluate!(AA, tmp, basis::PIBasis, Rs, Zs, z0)
   # compute the (multi-variate) density projection
   A = evaluate!(tmp.A, tmp.tmp_basis1p, basis.basis1p, Rs, Zs, z0)
   # now evaluate the correct inner basis, which doesn't know about z0, but
   # only ever sees A
   iz0 = z2i(basis, z0)
   evaluate!(AA, tmp, basis.inner[iz0], basis.evaluator, A)
   return @view(AA[1:length(basis.inner[iz0])])
end

# gradients

site_alloc_dB(basis::PIBasis, Rs::AbstractVector, args...) =
   site_alloc_dB(basis, length(Rs))

site_alloc_dB(basis::PIBasis, nmax::Integer) =
      zeros( JVec{fltype(basis)}, maximum(length.(basis.inner)), nmax )

alloc_temp_d(basis::PIBasis, Rs::AbstractVector, args...) =
   alloc_temp_d(basis, length(Rs))

alloc_temp_d(basis::PIBasis, nmax::Integer) =
      (
        A = alloc_B(basis.basis1p, nmax),
        dA = alloc_dB(basis.basis1p, nmax),
        tmp_basis1p = alloc_temp(basis.basis1p, nmax),
        tmpd_basis1p = alloc_temp_d(basis.basis1p, nmax),
        tmpd_inner = alloc_temp_d(basis.evaluator, basis, nmax)
      )


function evaluate_d!(AA, dAA, tmpd, basis::PIBasis, Rs, Zs, z0)
   iz0 = z2i(basis, z0)
   AAview = @view AA[basis.inner[iz0].AAindices]
   dAAview = @view dAA[basis.inner[iz0].AAindices, 1:length(Rs)]
   site_evaluate_d!(AAview, dAAview, tmpd, basis, Rs, Zs, z0)
   return dAA
end

function site_evaluate_d!(AA, dAA, tmpd, basis::PIBasis, Rs, Zs, z0)
   iz0 = z2i(basis, z0)
   # precompute the 1-p basis and its derivatives
   evaluate_d!(tmpd.A, tmpd.dA, tmpd.tmpd_basis1p, basis.basis1p, Rs, Zs, z0)
   site_evaluate_d!(AA, dAA, tmpd, basis.inner[iz0], basis.evaluator, tmpd.A, tmpd.dA)
   return nothing
end


#--------------------------------------------------------
# ------- specialised code for the standard evaluator

alloc_temp(::StandardEvaluator, basis::PIBasis) = nothing

alloc_temp_d(::StandardEvaluator, basis::PIBasis, args...) = nothing

# this method evaluates the InnerPIBasis, which is really the actual
# evaluation code; the rest is just bookkeeping
function evaluate!(AA, tmp, basis::InnerPIBasis, ::StandardEvaluator, A)
   fill!(AA, 1)
   for i = 1:length(basis)
      for α = 1:basis.orders[i]
         iA = basis.iAA2iA[i, α]
         AA[i] *= A[iA]
      end
   end
   return AA
end


function site_evaluate_d!(AA, dAA, tmpd, inner::InnerPIBasis,
                          _ev::StandardEvaluator, A, dA)
   # evaluate the AA basis
   evaluate!(AA, nothing, inner, _ev, tmpd.A)
   # loop over all neighbours
   for j = 1:size(dA, 2)
      # write the gradients into the correct slice of the dAA matrix
      for iAA = 1:length(inner)
         @inbounds dAA[iAA, j] = grad_AAi_Rj(iAA, j, inner, A, dA)
      end
   end
   return nothing
end



"""
Compute ∂∏A_a / ∂A_b = ∏_{a ≂̸ b} A_a
"""
function grad_AAi_Ab(iAA, b, inner, A)
   g = one(eltype(A))
   @inbounds for a = 1:inner.orders[iAA]
      if a != b
         g *= A[inner.iAA2iA[iAA, a]]
      end
   end
   return g
end

@doc raw"""
Compuate ∂∏A_a / ∂Rⱼ:
```math
   \frac{\partial \prod_a A_a}{\partial \bm r_j}
   =
   \sum_b \frac{\prod_{a} A_a}{\partial A_b} \cdot \frac{\partial A_b}{\partial \bm r_j}
   =
   \sum_b \prod_{a \neq b} A_a \cdot \frac{\partial \phi_b}{\partial \bm r_j}
```
"""
function grad_AAi_Rj(iAA, j, inner, A, dA)
   g = zero(eltype(dA))
   @inbounds for b = 1:inner.orders[iAA] # interaction order
      # A_{k_b} = A[iA]
      iA = inner.iAA2iA[iAA, b]
      # dAAi_dAb = ∂(∏A_{n_a}) / ∂A_{n_b}
      dAAi_dAb = grad_AAi_Ab(iAA, b, inner, A)
      g += dAAi_dAb * dA[iA, j]
   end
   return g
end


# --------------------------------------------------------
#   Evaluation code for DAG evaluator


alloc_temp(::DAGEvaluator, basis::PIBasis) =
   zeros(fltype(basis), maximum(inner.dag.numstore for inner in basis.inner))


function alloc_temp_d(::DAGEvaluator, basis::PIBasis,  nmax::Integer)
   maxstore = maximum(inner.dag.numstore for inner in basis.inner)
   return (
      AA = zeros(fltype(basis), maxstore),
      dAA = zeros(JVec{fltype(basis)}, maxstore, nmax)
   )
end

function evaluate!(AA, tmp, basis::InnerPIBasis, ::DAGEvaluator, A)
   AAdag = tmp.tmp_inner
   fill!(AAdag, 1)
   traverse_fwd!(AAdag, basis.dag, A,
                 (idx, AAval) -> if idx > 0; AA[idx] = AAval; end)
   return AA
end



function site_evaluate_d!(AA, dAA, tmpd, inner::InnerPIBasis,
                          ::DAGEvaluator, A, dA)
   dag = inner.dag
   nodes = dag.nodes
   idxs = dag.vals
   AAtmp = tmpd.tmpd_inner.AA
   dAAtmp = tmpd.tmpd_inner.dAA
   numneigs = size(dA, 2)
   # --- manual bounds checking
   @assert size(dAA, 2) >= numneigs
   @assert size(dAAtmp, 2) >= numneigs
   @assert length(AAtmp) >= inner.dag.numstore
   @assert size(dAAtmp, 1) >= inner.dag.numstore
   # ------------------

   @inline function _copyrow!(A_, iA, B_, iB)
      @simd for j = 1:numneigs
         @inbounds A_[iA,j] = B_[iB,j]
      end
   end

   @inline function _fwdgrad!(dAA_, i, n1, AA1, n2, AA2)
      @simd for j = 1:numneigs
         @inbounds dAA_[i, j] = AA1 * dAAtmp[n2,j] + AA2 * dAAtmp[n1,j]
      end
   end

   @inbounds for i = 1:dag.num1
      idx = idxs[i]
      AAtmp[i] = A[i]
      _copyrow!(dAAtmp, i, dA, i)
      if idx > 0
         AA[idx] = AAtmp[i]
         _copyrow!(dAA, idx, dAAtmp, i)
         # @. dAA[idx,:] = (@view dAAtmp[i,:])
      end
   end

   @inbounds for i = (dag.num1+1):dag.numstore
      n1, n2 = nodes[i]
      idx = idxs[i]
      AA1, AA2 = AAtmp[n1], AAtmp[n2]
      AAtmp[i] = AA1 * AA2
      _fwdgrad!(dAAtmp, i, n1, AA1, n2, AA2)
      # @. dAAtmp[i, :] = AA1 * (@view dAAtmp[n2,:]) + AA2 * (@view dAAtmp[n1,:])
      if idx > 0
         AA[idx] = AAtmp[i]
         _copyrow!(dAA, idx, dAAtmp, i)
         # @. dAA[idx, :] .= (@view dAAtmp[i, :])
      end
   end

   @inbounds for i = (dag.numstore+1):length(dag)
      idx = idxs[i]
      n1, n2 = nodes[i]
      # @assert idx > 0
      AA1, AA2 = AAtmp[n1], AAtmp[n2]
      AA[idx] = AA1 * AA2
      _fwdgrad!(dAA, idx, n1, AA1, n2, AA2)
      # @simd for j = 1:numneigs
      #    @inbounds dAA[idx, j] = AA1 * dAAtmp[n2,j] + AA2 * dAAtmp[n1,j]
      # end
   end

   return nothing
end
