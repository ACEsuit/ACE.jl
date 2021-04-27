
using SparseArrays: SparseMatrixCSC, sparse
using LinearAlgebra: mul!
using Combinatorics: permutations
using LinearAlgebra: rank, svd, Diagonal


"""
`struct SymmetricBasis`
"""
struct SymmetricBasis{BOP, PROP} <: ACEBasis
   pibasis::PIBasis{BOP}
   A2Bmap::SparseMatrixCSC{PROP, Int}
end


Base.length(basis::SymmetricBasis{BOP, PROP}) where {BOP, PROP} =
      size(basis.A2Bmap, 1)

fltype(basis::SymmetricBasis{BOP, PROP}) where {BOP, PROP} =  PROP
# rfltype(basis::SymmetricBasis) = rfltype(basis.pibasis)


SymmetricBasis(φ::AbstractProperty, args...; kwargs...) =
      SymmetricBasis(PIBasis(args...; kwargs...), φ)


function SymmetricBasis(pibasis, φ::TP) where {TP}

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
   #rotc = Rot3DCoeffs(rfltype(pibasis))
   rotc = Rot3DCoeffs(φ, rfltype(pibasis))
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
   return SymmetricBasis(pibasis, A2Bmap)
end


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
   rpebs = [ _nnllmm2b(nn, ll, mm) for mm in Ms ]

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


# function coupling_coeffs(bb, rotc::Rot3DCoeffs, φ::SphericalVector)
#    if length(bb) == 0
#       error("an equivariant vector basis function cannot have length 0")
#    end
#    ll, nn = _b2llnn(bb)
#    # A small modification here - the function yvec_symm_basis shall
#    # be φ related which specifies the blocks(the type of orbitals)...
#    U, Ms = Rotations3D.yvec_symm_basis(rotc, nn, ll, getL(φ))
#    rpibs = [ _nnllmm2b(nn, ll, mm) for mm in Ms ]
#    return U, rpibs
# end
#
# function coupling_coeffs(bb, rotc::Rot3DCoeffs, φ::SphericalMatrix)
#    if length(bb) == 0
#       error("an equivariant matrix basis function cannot have length 0")
#    end
#    ll, nn = _b2llnn(bb)
#    U, Ms = Rotations3D.mat_symm_basis(rotc, nn, ll, getL(φ)[1], getL(φ)[2])
#    rpibs = [ _nnllmm2b(nn, ll, mm) for mm in Ms ]
#    return U, rpibs
# end


_nnllmm2b(nn, ll, mm) = [ _nlm2b(n, l, m) for (n, l, m) in zip(nn, ll, mm) ]

@generated function _nlm2b(n::NamedTuple{KEYS}, l, m) where {KEYS}
   code = "b = (" * prod("$(k) = n.$(k), " for k in KEYS) * "l = l, m = m )"
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

using SparseArrays: AbstractSparseMatrixCSC, DenseInputVecOrMat,
				        nonzeros, rowvals, nzrange

function genmul!(C::StridedVecOrMat, A::AbstractSparseMatrixCSC, B::DenseInputVecOrMat, mulop)
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


# ---------------- Evaluation code

alloc_temp(basis::SymmetricBasis, args...) =
      (  AA = alloc_B(basis.pibasis),
         tmppi = alloc_temp(basis.pibasis) )

alloc_B(basis::SymmetricBasis, args...) =
      zeros(fltype(basis), length(basis))

function evaluate!(B, tmp, basis::SymmetricBasis,
                   cfg::AbstractConfiguration)
   # compute AA
   evaluate!(tmp.AA, tmp.tmppi, basis.pibasis, cfg)
   evaluate!(B, tmp, basis, tmp.AA)
   return B
end

# this function allows us to attach multiple symmetric bases to a single
#  𝑨 basis
#  TODO: this is extremely inefficient for multiple species But it is simple
#        and clean and will do for now...
function evaluate!(B, tmp, basis::SymmetricBasis,
                   AA::AbstractVector{<: Number})
   genmul!(B, basis.A2Bmap, AA, *)
end

# ---- gradients

function gradtype(basis::SymmetricBasis)
   φ = zero(eltype(basis.A2Bmap))
   dAA = zero(gradtype(basis.pibasis))
   return typeof(coco_o_daa(φ, dAA))
end

alloc_temp_d(basis::SymmetricBasis, nmax::Integer) =
      (  AA = alloc_B(basis.pibasis),
         dAA = alloc_dB(basis.pibasis, nmax),
         tmppi = alloc_temp(basis.pibasis),
         tmpdpi = alloc_temp_d(basis.pibasis, nmax) )

alloc_dB(basis::SymmetricBasis, nmax::Integer) =
      zeros(gradtype(basis), length(basis), nmax)

function evaluate_d!(dB, tmpd, basis::SymmetricBasis,
                     cfg::AbstractConfiguration)
   # compute AA
   evaluate_ed!(tmpd.AA, tmpd.dAA, tmpd.tmpdpi, basis.pibasis, cfg)
   evaluate_d!(dB, tmpd, basis, tmpd.AA, tmpd.dAA)
   return dB
end


function evaluate_d!(dB, tmpd, basis::SymmetricBasis,
                     AA::AbstractVector{<: Number}, dAA)
   genmul!(dB, basis.A2Bmap, dAA, ACE.coco_o_daa)
end
