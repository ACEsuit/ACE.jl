
using SparseArrays: SparseMatrixCSC, sparse
using LinearAlgebra: mul!
using Combinatorics: permutations
using LinearAlgebra: rank, svd, Diagonal


"""
`struct SymmetricBasis`

### Constructors

Option 1: pass a `PIBasis`
```julia
SymmetricBasis(pibasis, φ)
```
All possible permutation-invariant basis functions will be symmetrised and
then reduced to a basis (rather than spanning set)

Option 2: pass a `OneParticleBasis`
```julia
SymmetricBasis(φ, basis1p, maxν, maxdeg;
               Deg = NaiveTotalDegree())
```
will first construct a `PIBasis` from these inputs and then call the first
constructor.
"""
struct SymmetricBasis{BOP, PROP, REAL, VPROP} <: ACEBasis
   pibasis::PIBasis{BOP}
   A2Bmap::SparseMatrixCSC{PROP, Int}
   real::REAL
   B_pool::VectorPool{VPROP}
end

Base.length(basis::SymmetricBasis{BOP, PROP}) where {BOP, PROP} =
      size(basis.A2Bmap, 1)

valtype(basis::SymmetricBasis{BOP, PROP}) where {BOP, PROP} = basis.real(PROP)

# TODO: this is not nice, there should be proper promotion 
valtype(basis::SymmetricBasis{BOP, PROP}, X::AbstractState) where {BOP, PROP} = 
      valtype(basis)

gradtype(basis::SymmetricBasis, X::AbstractState) = gradtype(basis, typeof(X))

function gradtype(basis::SymmetricBasis, cfgorX)
   φ = zero(eltype(basis.A2Bmap))
   dAA = zero(gradtype(basis.pibasis, cfgorX))
   return typeof(_myreal1234( coco_o_daa(φ, dAA), basis.real))
end

# weird hacky name to avoid clashes 
# TODO: there must be a more elegant way to do this 
#       come to think of it, why did we do this in the first place???
_myreal1234(a, ::typeof(Base.identity)) = a
_myreal1234(a::StaticArray, ::typeof(Base.real)) = real.(a)

# -------- FIO

==(B1::SymmetricBasis, B2::SymmetricBasis) = 
      ( (B1.pibasis == B2.pibasis) && 
        (B1.A2Bmap == B2.A2Bmap) && 
        (B1.real == B2.real) )

write_dict(B::SymmetricBasis{BOP, PROP}) where {BOP, PROP} =
      Dict( "__id__" => "ACE_SymmetricBasis",
            "pibasis" => write_dict(B.pibasis),
            "A2Bmap" => write_dict(B.A2Bmap),
            "isreal" => (B.real == Base.real) )

read_dict(::Val{:ACE_SymmetricBasis}, D::Dict) =
      SymmetricBasis(read_dict(D["pibasis"]),
                     read_dict(D["A2Bmap"]),
                     (D["isreal"] ? Base.real : Base.identity) )
# --------

SymmetricBasis(φ::AbstractProperty, args...; isreal=false, kwargs...) =
      SymmetricBasis(PIBasis(args...; kwargs..., property = φ), φ; isreal=isreal)

function SymmetricBasis(pibasis, φ::TP; isreal=false) where {TP}

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

   # FUTURE: here we could analyze the different symbols and choose
   # which ones are of the "m"-type. e.g. there might be m1, m2 symbols. We
   # could create the convention that any symbol starting with m will be
   # treated like an m from the Ylm basis...

   # allocate the datastructure that computes and caches the
   # coupling coefficients
   # TODO: should this be stored with the basis?
   #       or maybe written to a file on disk? and then flushed every time
   #       we finish with a basis construction???
   rotc = Rot3DCoeffs(φ, real(valtype(pibasis)))
   # allocate triplet format
   Irow, Jcol, vals = Int[], Int[], TP[]
   # count the number of PI basis functions = number of rows
   idxB = 0

   # loop through AA basis, but skip most of them ...
   for (iAA, AA) in enumerate(AAspec)
      # AA = [b1, b2, ...], each bi = (n = ..., l = .., m = ...)
      # skip it unless all m are zero, because we want to consider each
      # (nn, ll, ...) block only once.
      # the loop over all possible `mm` must be taken care of inside
      # the `coupling_coeffs` implementation
      if !all(b.m == 0 for b in AA)
         continue
      end
      # get the coupling coefficients
      # here, we could help out a bit and make it easier to find the
      # the relevant basis functions?
      # AAcols will be in spec format i.e. named tuples
      U, AAcols = coupling_coeffs(AA, rotc)
      # loop over the rows of U -> each specifies a basis function
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
               @show bcol_ordered
               @show degree(bcol_ordered, NaiveTotalDegree(), pibasis.basis1p)
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
   return SymmetricBasis(pibasis, A2Bmap, isreal ? Base.real : Base.identity)
end

function SymmetricBasis(pibasis, A2Bmap, _real) 
   PROP = _real(eltype(A2Bmap))
   B_pool = VectorPool{PROP}() 
   return SymmetricBasis(pibasis, A2Bmap, _real, B_pool)
end


"""
produce the ordered tuple defining the AA basis function uniquely.
"""
function _get_ordered(bb, invAspec)
   iAs = [invAspec[b] for b in bb]
   return bb[ sortperm(iAs, rev = true) ]
end


function coupling_coeffs(bb, rotc::Rot3DCoeffs)
   # bb = [ b1, b2, b3, ...)
   # bi = (μ = ..., n = ..., l = ..., m = ...)
   #    (μ, n) -> n; only the l and m are used in the angular basis
   if length(bb) == 0
      # return [1.0,], [bb,]
		error("correlation order 0 is currently not allowed")
   end
   # convert to a format that the Rotations3D implementation can understand
   # this utility function splits the bb = (b1, b2 ...) with each
   # b1 = (μ = ..., n = ..., l = ..., m = ...) into
   #    l, and a new n = (μ, n)
   ll, nn = _b2llnn(bb)
   # now we can call the coupling coefficient construiction!!
   U, Ms = rpe_basis(rotc, nn, ll)

   # but now we need to convert the m spec back to complete basis function
   # specifications
   rpebs = [ _nnllmm2b(bb[1], nn, ll, mm) for mm in Ms ]

   return U, rpebs
end


function rpe_basis(A::Rot3DCoeffs,
						 nn::SVector{N, TN},
						 ll::SVector{N, Int}) where {N, TN}
	Ure, Mre = Rotations3D.re_basis(A, ll)
	G = _gramian(nn, ll, Ure, Mre)
   S = svd(G)
   rk = rank(Diagonal(S.S); rtol =  1e-7)
	Urpe = S.U[:, 1:rk]'
	return Diagonal(sqrt.(S.S[1:rk])) * Urpe * Ure, Mre
end


function _gramian(nn, ll, Ure, Mre)
   N = length(nn)
   nre = size(Ure, 1)
   G = zeros(Complex{Float64}, nre, nre)
   for σ in permutations(1:N)
      if (nn[σ] != nn) || (ll[σ] != ll); continue; end
      for (iU1, mm1) in enumerate(Mre), (iU2, mm2) in enumerate(Mre)
         if mm1[σ] == mm2
            for i1 = 1:nre, i2 = 1:nre
               G[i1, i2] += coco_dot(Ure[i1, iU1], Ure[i2, iU2])
            end
         end
      end
   end
   return G
end



_nnllmm2b(b, nn, ll, mm) = [ _nlm2b(b, n, l, m) for (n, l, m) in zip(nn, ll, mm) ]

@generated function _nlm2b(b::NamedTuple{ALLKEYS}, n::NamedTuple{NKEYS}, l, m) where {ALLKEYS, NKEYS}
   code =
      ( "( _b = (" * prod("$(k) = n.$(k), " for k in NKEYS) * "l = l, m = m ); "
         *
        " b = (" * prod("$(k) = _b.$(k), " for k in ALLKEYS) * ") )" )
   :( $(Meta.parse(code)) )
end


function _b2llnn(bb)
   @assert all( iszero(b.m) for b in bb )
   ll = SVector( [b.l for b in bb]... )
   nn = SVector( [_all_but_lm(b) for b in bb]... )
   return ll, nn
end

"""
return a names tuple containing all values in b except those corresponding
to l and m keys
"""
@generated function _all_but_lm(b::NamedTuple{NAMES}) where {NAMES}
   code = "n = ("
   for k in NAMES
      if !(k in [:l, :m])
         code *= "$(k) = b.$(k), "
      end
   end
   code *= ")"
   quote
      $(Meta.parse(code))
      n
   end
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

#dispatching for SVectors
#here we simply pass a coppy or a fill() or the mulop to every
#property on the SVector.
function genmul!(C::AbstractVector{<: SVector}, A::AbstractSparseMatrixCSC, B, mulop)
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
               mop = mulop(nzv[j], αxj)
               C[rv[j], k] += mop * ones(SVector{length(C[1]),eltype(mop)})
           end
       end
   end
   return C
end

# ---------------- Evaluation code


function evaluate!(B, basis::SymmetricBasis, cfg::AbstractConfiguration)
   # compute AA
   AA = acquire_B!(basis.pibasis, cfg)
   evaluate!(AA, basis.pibasis, cfg)
   return evaluate!(B, basis, AA)
end

# this function allows us to attach multiple symmetric bases to a single
#     𝑨 basis
#  TODO: this is extremely inefficient for multiple species But it is simple
#        and clean and will do for now...
function evaluate!(B, basis::SymmetricBasis, AA::AbstractVector{<: Number})
   genmul!(B, basis.A2Bmap, AA, (a, b) -> basis.real(a * b))
end


# ---------------- gradients

function evaluate_d!(dB, basis::SymmetricBasis, cfg::AbstractConfiguration)
   # compute AA
   AA = acquire_B!(basis.pibasis, cfg)
   dAA = acquire_dB!(basis.pibasis, cfg)
   evaluate_ed!(AA, dAA, basis.pibasis, cfg)
   evaluate_d!(dB, basis, AA, dAA)
   return dB
end


function evaluate_d!(dB, basis::SymmetricBasis,
                     AA::AbstractVector{<: Number}, dAA)
   genmul!(dB, basis.A2Bmap, dAA, 
           (a, b) -> _myreal1234(ACE.coco_o_daa(a, b), basis.real))
end

