

# -------------- Implementation of Product Basis

struct Product1PBasis{N, T <: Tuple}
   bases::T
   spec::Vector{NTuple{N, Int}}
end

Base.length(basis::Product1PBasis) = length(basis.spec)

outtype(basis::Product1PBasis) = promote_type(outtype.(basis.bases)...)

alloc_temp(basis::Product1PBasis) =
      (
         B = alloc_B.(basis.bases),
         tmp = alloc_temp.(basis.bases)
      )

function add_into_A!(A, tmp, basis::Product1PBasis{N}, Xj, Xi) where {N}
   Base.Cartesian.@nexprs N i -> evaluate!(tmp.B[i], tmp.tmp[i], basis.bases[i], Xj, Xi)
   for (iA, ϕ) in enumerate(basis.spec)
      t = one(eltype(A))
      Base.Cartesian.@nexprs N i -> (t *= tmp.B[i][ϕ[i]])
      A[iA] += t
   end
   return nothing
end
