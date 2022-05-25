
using SparseArrays: SparseMatrixCSC, sparse
using LinearAlgebra: mul!
using Combinatorics: permutations
using LinearAlgebra: rank, svd, Diagonal




"""
`struct SymmetricBasis`

### Constructors

Option 1: pass a `OneParticleBasis`
```julia
SymmetricBasis(φ, symgrp, basis1p, Bsel)
SymmetricBasis(φ, basis1p, Bsel)   # uses default symgrp = O3()
```
will first construct a `PIBasis` from these inputs and then call the second
constructor.


Option 1: pass a `PIBasis`
```julia
SymmetricBasis(φ, symgrp, pibasis)
SymmetricBasis(φ, pibasis)
```
If the PIbasis is already available, this directly constructs a 
resulting SymmetricBasis; all possible permutation-invariant basis functions 
will be symmetrised and then reduced to a basis (rather than spanning set)
"""
mutable struct SymmetricBasis{PIB, PROP, SYM, REAL} <: ACEBasis
   pibasis::PIB
   A2Bmap::SparseMatrixCSC{PROP, Int}
   symgrp::SYM
   real::REAL
end

Base.length(basis::SymmetricBasis{PIB, PROP}) where {PIB, PROP} =
      size(basis.A2Bmap, 1)

# -------- FIO

==(B1::SymmetricBasis, B2::SymmetricBasis) = 
      ( (B1.pibasis == B2.pibasis) && 
        (B1.A2Bmap == B2.A2Bmap) && 
        (B1.real == B2.real) )

write_dict(B::SymmetricBasis{PIB, PROP}) where {PIB, PROP} =
      Dict( "__id__" => "ACE_SymmetricBasis",
            "pibasis" => write_dict(B.pibasis),
            "A2Bmap" => write_dict(B.A2Bmap),
            "symgrp" => write_dict(B.symgrp), 
            "isreal" => (B.real == Base.real) )

read_dict(::Val{:ACE_SymmetricBasis}, D::Dict) =
      SymmetricBasis(read_dict(D["pibasis"]),
                     read_dict(D["A2Bmap"]),
                     read_dict(D["symgrp"]), 
                     (D["isreal"] ? Base.real : Base.identity) )
# --------

SymmetricBasis(φ::AbstractProperty, 
               basis1p::OneParticleBasis, 
               Bsel::AbstractBasisSelector; 
               kwargs...) =
      SymmetricBasis(φ, basis1p, O3(), Bsel; kwargs...)

SymmetricBasis(φ::AbstractProperty, pibasis; kwargs...) = 
      SymmetricBasis(φ, O3(), pibasis; kwargs...)


SymmetricBasis(φ::AbstractProperty, 
               basis1p::OneParticleBasis, 
               symgrp::SymmetryGroup, 
               Bsel::AbstractBasisSelector; 
               isreal=isrealB(φ), kwargs...) =
      SymmetricBasis(φ, symgrp, 
                     PIBasis(basis1p, symgrp, Bsel; 
                             isreal = isrealAA(φ), kwargs..., property = φ); 
                     isreal=isreal)

SymmetricBasis(φ::AbstractProperty, symgrp::SymmetryGroup, pibasis::PIBasis; 
               isreal=false) = 
      SymmetricBasis(φ, symgrp, pibasis, isreal ? Base.real : Base.identity)


function SymmetricBasis(φ::TP, symgrp::SymmetryGroup, pibasis::PIBasis, 
                        _real) where {TP <: AbstractProperty}

   # AA index -> AA spec
   AAspec = get_spec(pibasis)

   # construct the reverse mapping AA spec -> iAA and A spec -> iA
   invAAspec = Dict{Any, Int}()
   for (iAA, AA) in enumerate(AAspec)
      invAAspec[AA] = iAA
   end
   invAspec = Dict{Any, Int}()
   for (iA, A) in enumerate(get_spec(pibasis.basis1p))
      invAspec[A] = iA
   end

   # allocate the datastructure that computes and caches the
   # coupling coefficients
   # TODO: should this be stored with the basis?
   #       or maybe written to a file on disk? and then flushed every time
   #       we finish with a basis construction???
   # TODO: for sure this needs to become a function of the symmetry group?
   rotc = Rot3DCoeffs(φ, Float64)
   # allocate triplet format
   TCC = coco_type(TP)
   Irow, Jcol, vals = Int[], Int[], TCC[]
   # count the number of PI basis functions = number of rows
   idxB = 0

   # loop through AA basis, but skip most of them ...
   for (iAA, AA) in enumerate(AAspec)
      # determine whether we need to compute coupling coefficients for this 
      # basis function or whether it will be included in a different 
      # coco computation? 
      if !is_refbasisfcn(symgrp, AA)
         continue 
      end 
      # compute the cocos 
      U, AAcols = coupling_coeffs(symgrp, AA, rotc)

      # loop over the rows of U -> each specifies a basis function which we now 
      # need to incorporate into the basis
      for irow = 1:size(U, 1)
         idxB += 1
         # loop over the columns of U / over brows
         for (icol, bcol) in enumerate(AAcols)
            # put bcol into the correct order
            bcol_ordered = _get_ordered(bcol, invAspec)
            # this is a subtle step: bcol and bcol_ordered are equivalent
            # permutation-invariant basis functions. This means we will
            # add the same PI basis function several times, but in the call to
            # `sparse` the values will just be added.
            if !haskey(invAAspec, bcol_ordered)
               @warn("bcol_ordered not in AA-spec")
               @show bcol_ordered
               # @show degree(bcol_ordered, NaiveTotalDegree(), pibasis.basis1p)
               error("bcol_ordered not in AA-spec")
            end
            idxAA = invAAspec[bcol_ordered]
            push!(Irow, idxB)
            push!(Jcol, idxAA)
            push!(vals, U[irow, icol])
         end
      end
   end
   # TODO: filter and throw out everything that hasn't been used!!
   # create CSC: [   triplet    ]  nrows   ncols
   A2Bmap = sparse(Irow, Jcol, vals, idxB, length(AAspec))
   basis = SymmetricBasis(pibasis, A2Bmap, symgrp, _real)
   # clean up a bit, i.e. remove AA basis functions that we don't need
   # to evaluate the symmetric basis
   clean_pibasis!(basis)
   return basis
end

function SymmetricBasis(pibasis, A2Bmap, symgrp, _real) 
   PROP = _real(eltype(A2Bmap))
   B_pool = VectorPool{PROP}() 
   return SymmetricBasis(pibasis, A2Bmap, symgrp, _real, B_pool)
end


"""
produce the ordered tuple defining the AA basis function uniquely.
"""
function _get_ordered(bb, invAspec)
   iAs = [invAspec[b] for b in bb]
   return bb[ sortperm(iAs, rev = true) ]
end


# ------------------- exporting the basis spec 

# this doesn't provide the "full" specification, just collects 
# the n and l but not the m or coupling coefficients. 

function get_spec(basis::SymmetricBasis)
   spec_AA = get_spec(basis.pibasis)
   spec_B = [] 
   for iB = 1:length(basis)
      iAA = findfirst(norm.(basis.A2Bmap[iB, :]) .!= 0)
      push!(spec_B, get_sym_spec(basis.symgrp, spec_AA[iAA]))
   end
   return identity.(spec_B)
end


# ---------------- sparsification and cleanup codes 

"""
`sparsify!(basis::SymmetricBasis; del = ..., keep = ...)`

sparsify the symmetric basis by either specifyin which basis functions to 
keep or which ones to delete.
"""
function sparsify!(basis::SymmetricBasis; 
                  del::Union{Nothing, AbstractVector{<: Integer}} = nothing, 
                  keep::Union{Nothing, AbstractVector{<: Integer}} = nothing)
   if ( (del == nothing && keep == nothing ) || 
        (del != nothing && keep != nothing ) )
      error("sparsify!: must provide either del or keep kwarg but not both")
   end 
   if del != nothing 
      keep = setdiff(1:length(basis), del)
   end
   basis.A2Bmap = basis.A2Bmap[keep, :]
   clean_pibasis!(basis; atol = 0.0)
end

# this is also used for Filtering the AA basis if there are zero-rows in the A2B map

"""
Remove elements of the AA basis, when there are zero-rows in the A2B map, i.e.
when some AA basis elements are simply not required to evaluate the 
symmetric basis. 
"""
function clean_pibasis!(basis::SymmetricBasis; atol = 0.0)
   # get the zero-columns 
   Inz = sort( findall(x -> x > atol, sum(norm, basis.A2Bmap, dims=1)[:]) )
   if length(Inz) < size(basis.A2Bmap, 2)
      # remove those columns from the A2Bmap 
      sparsify!(basis.pibasis, Inz)
      # remove those columns from the A2Bmap 
      basis.A2Bmap = basis.A2Bmap[:, Inz] 
   end
   clean_1pbasis!(basis.pibasis)
   return basis
end

# ---------------- A modified sparse matmul

# TODO: move this stuff all to aux? 

using SparseArrays: AbstractSparseMatrixCSC,
				        nonzeros, rowvals, nzrange

using LinearAlgebra: Transpose


function genmul!(C, A::AbstractSparseMatrixCSC, B, mulop)
    size(A, 2) == size(B, 1) || throw(DimensionMismatch())
    size(A, 1) == size(C, 1) || throw(DimensionMismatch())
    size(B, 2) == size(C, 2) || throw(DimensionMismatch())
    nzv = nonzeros(A)
    rv = rowvals(A)
    fill!(C, zero(eltype(C)))
    for k in 1:size(C, 2)
        @inbounds for col in 1:size(A, 2)
            αxj = B[col,k]
            for j in nzrange(A, col)
                C[rv[j], k] += mulop(nzv[j], αxj)
            end
        end
    end
    return C
end


function genmul!(C, xA::Transpose{<:Any,<:AbstractSparseMatrixCSC}, B, mulop)
   A = xA.parent
   size(A, 2) == size(C, 1) || throw(DimensionMismatch())
   size(A, 1) == size(B, 1) || throw(DimensionMismatch())
   size(B, 2) == size(C, 2) || throw(DimensionMismatch())
   nzv = nonzeros(A)
   rv = rowvals(A)
   fill!(C, zero(eltype(C)))
   for k in 1:size(C, 2)
       @inbounds for col in 1:size(A, 2)
           tmp = zero(eltype(C))
           for j in nzrange(A, col)
               tmp += mulop(nzv[j], B[rv[j],k])
           end
           C[col,k] += tmp
       end
   end
   return C
end


function genmul(A, B, mulop)
   T = typeof(mulop(A[1], B[1]))
   C = Array{T}(undef, (size(A, 1), size(B)[2:end]...) )
   return genmul!(C, A, B, mulop)
end


# ---------------- Evaluation code

function evaluate(basis::SymmetricBasis, cfg::UConfig)
   AA = evaluate(basis.pibasis, cfg)
   B = evaluate(basis, AA)
   release!(AA)
   return B 
end

# NOTE: Nasty and completely not understood type instability here 
function evaluate!(B, basis::SymmetricBasis, cfg::UConfig)
   AA = evaluate(basis.pibasis, cfg)
   evaluate!(B, basis, AA)
   release!(AA)
   return B 
end

evaluate(basis::SymmetricBasis, AA::AbstractVector{<: Number}) = 
      genmul(basis.A2Bmap, AA, (a, b) -> basis.real(a * b))

evaluate!(B, basis::SymmetricBasis, AA::AbstractVector{<: Number}) = 
      genmul!(B, basis.A2Bmap, AA, (a, b) -> basis.real(a * b))


# ---------------- gradients

# args... could be nothing or sym
function evaluate_ed(basis::SymmetricBasis, cfg::UConfig, args...)
   AA, dAA = evaluate_ed(basis.pibasis, cfg, args...)
   B, dB = evaluate_ed(basis, AA, dAA) 
   release!(AA)
   release!(dAA)
   return B, dB
end

function evaluate_ed(basis::SymmetricBasis, AA::AbstractVector{<: Number}, dAA)
   B = genmul(basis.A2Bmap, AA, (a, b) -> basis.real(a * b))
   dB = genmul(basis.A2Bmap, dAA, (a, b) -> basis.real( ACE.coco_o_daa(a, b) ))
   return B, dB 
end

evaluate_d(basis::SymmetricBasis, args...) = evaluate_ed(basis, args...)[2]

# ------------------------------- 


function scaling(basis::SymmetricBasis, p)
   wwpi = scaling(basis.pibasis, p)
   wwrpi = abs2.(norm.(basis.A2Bmap)) * abs2.(wwpi)
   return sqrt.(wwrpi)
end


