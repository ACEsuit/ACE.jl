# --------------------------------------------------------------------------
# ACE.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


using SparseArrays: SparseMatrixCSC, sparse
using LinearAlgebra: mul!



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
   rotc = Rot3DCoeffs(rfltype(pibasis))

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
      U, AAcols = coupling_coeffs(AA, rotc, φ)

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
            push!(vals, TP(U[irow, icol]))
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


function coupling_coeffs(bb, rotc::Rot3DCoeffs, φ::Invariant)
   # bb = [ b1, b2, b3, ...)
   # bi = (μ = ..., n = ..., l = ..., m = ...)
   #    (μ, n) -> n; only the l and m are used in the angular basis
   if length(bb) == 0
      return [1.0,], [bb,]
   end
   # convert to a format that the Rotations3D implementation can understand
   # this utility function splits the bb = (b1, b2 ...) with each
   # b1 = (μ = ..., n = ..., l = ..., m = ...) into
   #    l, and a new n = (μ, n)
   ll, nn = _b2llnn(bb)

   # now we can call the coupling coefficient construiction!!
   U, Ms = Rotations3D.rpi_basis(rotc, nn, ll)

   # but now we need to convert the m spec back to complete basis function
   # specifications
   rpibs = [ _nnllmm2b(nn, ll, mm) for mm in Ms ]

   return U, rpibs
end

function coupling_coeffs(bb, rotc::Rot3DCoeffs, φ::EuclideanVector)
   if length(bb) == 0
      error("an equivariant vector basis function cannot have length 0")
   end
   ll, nn = _b2llnn(bb)
   U, Ms = Rotations3D.vec3_symm_basis(rotc, nn, ll)
   rpibs = [ _nnllmm2b(nn, ll, mm) for mm in Ms ]
   return U, rpibs
end

function coupling_coeffs(bb, rotc::Rot3DCoeffs, φ::SphericalVector)
   if length(bb) == 0
      error("an equivariant vector basis function cannot have length 0")
   end
   ll, nn = _b2llnn(bb)
   # A small modification here - the function yvec_symm_basis shall
   # be φ related which specifies the blocks(the type of orbitals)...
   U, Ms = Rotations3D.yvec_symm_basis(rotc, nn, ll, φ)
   rpibs = [ _nnllmm2b(nn, ll, mm) for mm in Ms ]
   return U, rpibs
end


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


# ---------------- Evaluation code

alloc_temp(basis::SymmetricBasis) =
      (  AA = alloc_B(basis.pibasis),
         tmppi = alloc_temp(basis.pibasis) )

alloc_B(basis::SymmetricBasis) =
      zeros(fltype(basis), length(basis))

function evaluate!(B, tmp, basis::SymmetricBasis,
                   Xs::AbstractVector{<: AbstractState}, X0::AbstractState)
   # compute AA
   evaluate!(tmp.AA, tmp.tmppi, basis.pibasis, Xs, X0)
   evaluate!(B, tmp, basis, tmp.AA)
   return B
end

# this function allows us to attach multiple symmetric bases to a single
#  𝑨 basis
#  TODO: this is extremely inefficient for multiple species But it is simple
#        and clean and will do for now...
function evaluate!(B, tmp, basis::SymmetricBasis,
                   AA::AbstractVector{<: Number})
   mul!(B, basis.A2Bmap, AA)
end
