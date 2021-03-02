

using SparseArrays: SparseMatrixCSC, sparse
using LinearAlgebra: mul!



"""
`struct SymmetricBasis`
"""
struct SymmetricBasis{BOP, PROP} <: IPBasis
   pibasis::PIBasis{BOP}
   A2Bmap::SparseMatrixCSC{PROP, Int}
end


Base.length(basis::SymmetricBasis{T}) where {T} =
      length(zero(T)) * size(basis.A2Bmaps, 1)

# fltype(basis::SymmetricBasis) = fltype(basis.pibasis)
# rfltype(basis::SymmetricBasis) = rfltype(basis.pibasis)

# TODO: move into atoms stuff
cutoff(basis::SymmetricBasis) = cutoff(basis.pibasis)



function SymmetricBasis(pibasis, φ::TP) where {TP}

   # AA index -> AA spec
   AAspec = get_spec(pibasis)

   # construct the reverse mapping AA spec -> iAA
   invAAspec = Dict()
   for (iAA, AA) in AAspec
      invAAspec[AA] = iAA
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
      # skip it unless all m are zero, because we want to consider each
      # (nn, ll, ...) block only once.
      if !all(b.m == 0 for b in AA)
         continue
      end
      # get the coupling coefficients
      # here, we could help out a bit and make it easier to find the
      # the relevant basis functions?
      U, AAcols = coupling_coeffs(AAspec, invAAspec, rotc, AA, φ)

      # loop over the rows of U -> each specifies a basis function
      for irow = 1:size(U, 1)
         idxB += 1
         # loop over the columns of U / over brows
         for (icol, bcol) in enumerate(AAcols)
            # this is a subtle step: bcol and bcol_ordered are equivalent
            # permutation-invariant basis functions. This means we will
            # add the same PI basis function several times, but in the call to
            # `sparse` the values will just be added.
            idxAA = _get_AA_index(invAAspec, bcol)
            push!(Irow, idxB)
            push!(Jcol, idxAA)
            push!(vals, U[irow, icol])
         end
      end
   end
   # TODO: filter and throw out everything that hasn't been used!!
   # create CSC: [   triplet    ]  nrows   ncols
   return sparse(Irow, Jcol, vals, idxB, length(AAspec))
end



function coupling_coeffs(AAspec, invAAspec, rotc::Rot3DCoeffs, bb, φ::Invariant)
   if length(bb) == 0
      error("TODO: implement the constant case?")
   end
   # convert to a format that the Rotations3D implementation can understand
   # this utility function splits the bb = (b1, b2 ...) with each
   # b1 = (μ = ..., n = ..., l = ..., m = ...) into
   #    l, and a new n = (μ, n)
   ll, nn = get_ll_nn(bb)

   # now we can call the coupling coefficient construiction!!
   U, Ms = Rotations3D.rpi_basis(rotc, zz, nn, ll)

   # but finally we need to convert the indices back to basis function
   # specifications
   rpibs = [ _znlms2b(zz, nn, ll, mm, pib.z0) for mm in Ms ]

   return U, rpibs
end

function get_ll_nn(bb)
   ll = SVector( [b.l for b in bb]... )
   nn = SVector( [_all_but_lm(b) for b in bb]... )
   return ll, nn
end

"""
return a tuple containing all values in b except those corresponding
to l and m keys
"""
function _all_but_lm(b)
   n = Int[]
   for k in keys(b)
      if !(k in [:l, :m])
         push!(n, b[k])
      end
   end
   return tuple(n...)
end




# ---------------- Evaluation code


function evaluate!(B, tmp, basis::SymmetricBasis,
                   Xs::AbstractVector{<: AbstractState}, X0::AbstractState)
   # compute AA
   evaluate!(tmp.AA, tmp.tmppi, basis.pibasis, Xs, X0)
   evaluate!(B, tmp, basis, AA)
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
