
using JuLIP: evaluate

# ------- EndlessVector
#   auxiliary datastructure wrapping sparse vectors

struct EndlessVector{T}
   x::SparseVector{T, Int}
end

function endless(x::SparseVector{T, Int}; prune=true, tol = 1e-10) where {T}
   if prune
      x = droptol!(x, tol)
      x = x[1:maximum(x.nzind)]
   end
   return EndlessVector(x)
end

endless(_x::AbstractVector; kwargs...) = endless(sparse(_x); kwargs...)

Base.getindex(x::EndlessVector{T}, i) where {T} =
      (0 < i <= length(x.x)) ? x.x[i] : zero(T)

Base.length(x::EndlessVector) = length(x.x)

# ------- OrthPolyProdCoeffs
#   the main data structure for storing product coefficients for the
#   radial basis

mutable struct OrthPolyProdCoeffs{T}
   basis::OrthPolyBasis{T}
   coeffs::Dict{Tuple{Int, Int}, EndlessVector{T}}
end

OrthPolyProdCoeffs(basis::OrthPolyBasis{T}) where {T} =
   OrthPolyProdCoeffs(basis, Dict{Tuple{Int, Int}, EndlessVector{T}}())


function (coeffs::OrthPolyProdCoeffs{T})(n1, n2) where {T}
   n1, n2 = extrema((n1, n2))  # n1 <= n2
   if n1 <= 0
      return endless(T[])
   end
   if !haskey(coeffs.coeffs, (n1, n2))
      coeffs.coeffs[(n1, n2)] = _precompute_prodcoeffs(coeffs, n1, n2)
   end
   return coeffs.coeffs[(n1, n2)]
end


function _precompute_prodcoeffs(coeffs::OrthPolyProdCoeffs{T}, n1, n2) where {T}
   # we want to expand this function in Jn basis
   basis = coeffs.basis
   f(x) = (J = evaluate(basis, x); J[n1] * J[n2])
   # evaluate basis function with index ν
   evalJ(x, ν) = evaluate(basis, x)[ν]
   # get the inner product information, normalise the weights s.t. <1, 1> = 1
   # TODO: abstract this out!
   tdf = basis.tdf
   ww = basis.ww
   ww = ww / sum(ww)
   dotJ(f1, f2) = dot(f1.(tdf), ww .* f2.(tdf))

   # now we can get the coefficients (note the basis is orthonormal!!)
   maxn = (n1 + n2 - 2) + (basis.pl + basis.pr + 1)
   P = [ dotJ(f, x -> evalJ(x, ν)) for ν = 1:maxn ]
   return endless(P)
end
