
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


export PIBasis

"""
`PIBasisFcn{N, TOP}` : represents a single multivariate basis function
in terms of 1-particle pasis functions in each coordinate direction. Crucially,
this function will be interpreted as a *permutation invariant* basis function!
"""
struct PIBasisFcn{N, TOP <: OnepBasisFcn}
   z0::AtomicNumber
   oneps::NTuple{N, TOP}
end

PIBasisFcn(z0::AtomicNumber, oneps::AbstractVector) =
   PIBasisFcn(z0, tuple(oneps...))

order(b::PIBasisFcn{N}) where {N} = N

degree(d::AbstractDegree, pphi::PIBasisFcn) = degree(d, pphi.oneps)

# TODO: this is very rough - can we do better?
scaling(b::PIBasisFcn, p) = sum(scaling(bb, p) for bb in b.oneps)

# TODO: can we replace this with get_basis_spec?
function PIBasisFcn(Aspec, t, z0::AtomicNumber)
   if isempty(t) || sum(abs, t) == 0
      return PIBasisFcn{0, eltype(Aspec)}(z0, tuple())
   end
   # zeros stand for reduction in body-order
   tnz = t[findall(t .!= 0)]
   return PIBasisFcn(z0, Aspec[[tnz...]])
end

write_dict(b::PIBasisFcn) =
   Dict("__id__" => "SHIPs_PIBasisFcn",
        "z0" => write_dict(b.z0),
        "oneps" => write_dict.(b.oneps))

read_dict(::Val{:SHIPs_PIBasisFcn}, D::Dict) =
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
   Aspec_p = sort(Aspec, by = degree)
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


abstract type AbstractInnerPIBasis <: IPBasis end

"""
`mutable struct InnerPIBasis` : this type is just an auxilary type to
make the implementation of `PIBasis` clearer. It implements the
permutation-invariant basis for a single centre-atom species. The main
type `PIBasis` then stores `NZ` objects of type `InnerPIBasis`
and "dispatches" the work accordingly.
"""
mutable struct InnerPIBasis <: AbstractInnerPIBasis
   orders::Vector{Int}           # order (length) of ith basis function
   iAA2iA::Matrix{Int}           # where in A can we find the ith basis function
   b2iAA::Dict{PIBasisFcn, Int}  # mapping PIBasisFcn -> iAA =  index in AA[z0]
   b2iA::Dict{Any, Int}          # mapping from 1-p basis fcn to index in A[z0]
   AAindices::UnitRange{Int}     # where in AA does AA[z0] fit?
end

==(B1::InnerPIBasis, B2::InnerPIBasis) = (
   (B1.b2iA == B2.b2iA) &&
   (B1.iAA2iA == B2.iAA2iA) )

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

   # put it all together ...
   return InnerPIBasis(orders, iAA2iA, b2iAA, b2iA, AAindices)
end







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
mutable struct PIBasis{BOP, NZ, TIN} <: IPBasis
   basis1p::BOP             # a one-particle basis
   zlist::SZList{NZ}
   inner::NTuple{NZ, TIN}
end

cutoff(basis::PIBasis) = cutoff(basis.basis1p)

==(B1::PIBasis, B2::PIBasis) = SHIPs._allfieldsequal(B1, B2)

Base.eltype(basis::PIBasis) = eltype(basis.basis1p)

Base.length(basis::PIBasis) = sum(length(basis, iz) for iz = 1:numz(basis))
Base.length(basis::PIBasis, iz0::Integer) = length(basis.inner[iz0])
Base.length(basis::PIBasis, z0::AtomicNumber) = length(basis, z2i(basis, z0))

function PIBasis(basis1p::OneParticleBasis,
                 N::Integer,
                 D::AbstractDegree,
                 maxdeg::Real;
                 filter = b -> order(b) > 0,
                 evaluator = :classic)
   innerspecs = Any[]
   # now for each iz0 (i.e. for each z0) construct an "inner basis".
   for iz0 = 1:numz(basis1p)
      z0 = i2z(basis1p, iz0)
      # get a list of 1-p basis function
      AAspec_z0 = get_PI_spec(basis1p, N, D, maxdeg, z0; filter=filter)
      push!(innerspecs, AAspec_z0)
   end
   return pibasis_from_specs(basis1p, identity.(innerspecs),
                             _evaluator(evaluator))
end

function _evaluator(evaluator)
   if evaluator == :classic
      return InnerPIBasis
   elseif evaluator == :dag
      return DAGInnerPIBasis
   end
   error("Unknown evaluator")
end


# TODO: instead of copying zlist, maybe forward the zlist methods

function pibasis_from_specs(basis1p, innerspecs, InnerType)
   idx = 0
   inner = InnerType[]
   for iz0 = 1:numz(basis1p)
      z0 = i2z(basis1p, iz0)
      Aspec_z0 = get_basis_spec(basis1p, z0)
      AAspec_z0 = innerspecs[iz0]
      AAindices = (idx+1):(idx+length(AAspec_z0))
      push!(inner, InnerType(Aspec_z0, AAspec_z0, AAindices, z0))
      idx += length(AAspec_z0)
   end
   return PIBasis(basis1p, basis1p.zlist, tuple(inner...))
end


"""
`get_basis_spec(basis::PIBasis, iz0::Integer, i::Integer)`

Here `i` is the index of the basis function for which we reconstruct its
specification.
"""
function get_basis_spec(basis::PIBasis, iz0::Integer, i::Integer)
   N = basis.inner[iz0].orders[i]
   iAA2iA = basis.inner[iz0].iAA2iA[i, 1:N]
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


# -------------------------------------------------
# FIO codes


write_dict(basis::PIBasis) =
   Dict(  "__id__" => "SHIPs_PIBasis",
         "basis1p" => write_dict(basis.basis1p),
           "inner" => [ write_dict.( collect(keys(basis.inner[iz0].b2iAA)) )
                         for iz0 = 1:numz(basis) ],
       "evaluator" => "classic")

function read_dict(::Val{:SHIPs_PIBasis}, D::Dict)
   basis1p = read_dict(D["basis1p"])
   innerspecs = [ read_dict.(D["inner"][iz0])  for iz0 = 1:numz(basis1p) ]
   return pibasis_from_specs(basis1p, innerspecs,
                             _evaluator(Symbol(D["evaluator"])))
end


# -------------------------------------------------
# Evaluation codes

site_alloc_B(basis::PIBasis, args...) =
      zeros( eltype(basis), maximum(length.(basis.inner)) )

alloc_temp(basis::PIBasis, args...) =
      ( A = alloc_B(basis.basis1p, args...),
        tmp_basis1p = alloc_temp(basis.basis1p, args...),
        tmp_inner = alloc_temp(typeof(basis.inner[1]), basis)
      )

alloc_temp(::Type{InnerPIBasis}, basis::PIBasis) = nothing

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
   evaluate!(AA, tmp, basis.inner[iz0], A)
   return nothing; # @view(AA[1:length(basis.inner[iz0])])
end

# this method evaluates the InnerPIBasis, which is really the actual
# evaluation code; the rest is just bookkeeping
function evaluate!(AA, tmp, basis::InnerPIBasis, A)
   fill!(AA, 1)
   for i = 1:length(basis)
      for α = 1:basis.orders[i]
         iA = basis.iAA2iA[i, α]
         AA[i] *= A[iA]
      end
   end
   return AA
end


# --------------------------------------------------------
# gradients: this section of functions is for computing
#  ∂∏A / ∂R_j


site_alloc_dB(basis::PIBasis, args...) =
      zeros( JVec{eltype(basis)}, maximum(length.(basis.inner)) )

alloc_temp_d(basis::PIBasis, args...) =
      (
        A = alloc_B(basis.basis1p, args...),
        dA = alloc_dB(basis.basis1p, args...),
        tmp_basis1p = alloc_temp(basis.basis1p, args...),
        tmpd_basis1p = alloc_temp_d(basis.basis1p, args...)
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
   site_evaluate_d!(AA, dAA, tmpd, basis.inner[iz0], tmpd.A, tmpd.dA)
   return nothing
end

function site_evaluate_d!(AA, dAA, tmpd, inner::InnerPIBasis, A, dA)
   # evaluate the AA basis
   evaluate!(AA, nothing, inner, tmpd.A)
   # loop over all neighbours
   for j = 1:size(dA, 2)
      # write the gradients into the correct slice of the dAA matrix
      dAAj = @view dAA[:, j]
      evaluate_d_Rj!(dAAj, inner, A, dA, j)
   end
   return nothing
end



"""
Compute ∂∏A_a / ∂A_b = ∏_{a ≂̸ b} A_a
"""
function grad_AAi_Ab(iAA, b, inner, A)
   g = one(eltype(A))
   for a = 1:inner.orders[iAA]
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
   for b = 1:inner.orders[iAA] # interaction order
      # A_{k_b} = A[iA]
      iA = inner.iAA2iA[iAA, b]
      # dAAi_dAb = ∂(∏A_{n_a}) / ∂A_{n_b}
      dAAi_dAb = grad_AAi_Ab(iAA, b, inner, A)
      g += dAAi_dAb * dA[iA, j]
   end
   return g
end


"""
evaluate ∂AA / ∂Rⱼ
"""
function evaluate_d_Rj!(dAAj, inner::InnerPIBasis, A, dA, j) where {T}
   for iAA = 1:length(inner)
      dAAj[iAA] = grad_AAi_Rj(iAA, j, inner, A, dA)
   end
   return dAAj
end

# """
# evaluate ∂AA[z0] / ∂Rⱼ
# """
# evaluate_d_Rj!(dAAj, pibasis::PIBasis, A, dA, z0, j) =
#       evaluate_d_Rj!(dAAj, pibasis.inner[z2i(pibasis, z0)], A, dA, j)



# --------------------------------------------------------
#   Alternative InnerPIBasis based on the graph-evaluator

include("grapheval.jl")

import SHIPs.DAG: CorrEvalGraph, get_eval_graph, traverse_dag!

mutable struct DAGInnerPIBasis <: AbstractInnerPIBasis
   b1pspec::Vector{Any}
   dag::CorrEvalGraph{Int, Int}
   len::Int
   z0::AtomicNumber
   AAindices::UnitRange{Int}
end

Base.length(inner::DAGInnerPIBasis) = inner.len

function DAGInnerPIBasis(Aspec::AbstractVector, AAspec::AbstractVector,
                         AAindices, z0; kwargs...)
   classic = InnerPIBasis(Aspec, AAspec, AAindices, z0)
   len = length(classic)
   dag = get_eval_graph(classic, collect(1:len); kwargs...)
   return DAGInnerPIBasis(Aspec, dag, len, z0, classic.AAindices)
end


alloc_temp(::Type{DAGInnerPIBasis}, basis::PIBasis) =
   zeros(eltype(basis), maximum(inner.dag.numstore for inner in basis.inner))


function evaluate!(AA, tmp, basis::DAGInnerPIBasis, A)
   AAdag = tmp.tmp_inner
   fill!(AAdag, 1)
   traverse_dag!(AAdag, basis.dag, A,
                 (idx, AAval) -> if idx > 0; AA[idx] = AAval; end)
   return AA
end



# function site_evaluate_d!(AA, dAA, tmpd, inner::InnerPIBasis, A, dA)
#    # evaluate the AA basis
#    evaluate!(AA, nothing, inner, tmpd.A)
#    # loop over all neighbours
#    for j = 1:size(dA, 2)
#       # write the gradients into the correct slice of the dAA matrix
#       dAAj = @view dAA[:, j]
#       evaluate_d_Rj!(dAAj, inner, A, dA, j)
#    end
#    return nothing
# end


function site_evaluate_d!(AA, dAA, tmpd, inner::DAGInnerPIBasis, A, dA)
   nodes = inner.dag.nodes
   idxs = inner.dag.vals
   AAtmp = tmpd.tmpd_inner.AA
   dAAtmp = tmpd.tmpd_inner.dAA

   for i = 1:dag.num1
      idx = idxs[i]
      AAtmp[i] = A[i]
      @. dAAtmp[i, :] = dA[i, :]
      if idx > 0
         AA[idx] = AAtmp[i]
         @. dAA[idx,:] = dAAtmp[i,:]
      end
   end

   for i = (dag.num1+1):dag.numstore
      n1, n2 = nodes[i]
      idx = idxs[i]
      AA1, AA2 = AAtmp[n1], AAtmp[n2]
      AAtmp[i] = AA1 * AA2
      @. dAAtmp[i, :] = AA1 * dAAtmp[n2,:] + AA2 * dAAtmp[n1,:]
      if idx > 0
         AA[idx] = AAtmp[i]
         @. dAA[idx, :] = dAAtmp[i, :]
      end
   end

   for i = (dag.num1+1):dag.numstore
      idx = idxs[i]
      n1, n2 = nodes[i]
      @assert idx > 0
      AA1, AA2 = AAtmp[n1], AAtmp[n2]
      AA[idx] = AA1 * AA2
      @. dAA[idx, :] = AA1 * dAAtmp[n2,:] + AA2 * dAAtmp[n1,:]
   end

   return nothing
end
