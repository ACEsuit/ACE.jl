
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
# gradients

gradtype(basis::PIBasis, cfgorX) = basis.real( gradtype(basis.basis1p, cfgorX) )

alloc_dB(basis::PIBasis, cfg::AbstractConfiguration, nmax = length(cfg)) =
      zeros(gradtype(basis, cfg), (length(basis), nmax))

alloc_temp_d(basis::PIBasis, cfg::AbstractConfiguration, nmax = length(cfg)) =
      (
        A = alloc_B(basis.basis1p),
        dA = alloc_dB(basis.basis1p, cfg),
        tmp_basis1p = alloc_temp(basis.basis1p),
        tmpd_basis1p = alloc_temp_d(basis.basis1p, cfg),
        # ---- adjoint stuff
        dAAdA = zeros(fltype(basis.basis1p),
                      maximum(basis.spec.orders))
      )

# TODO: This is a naive forwardmode implementation; we need
#       to switch this to reverse mode differentiation

function evaluate_ed!(AA, dAA, tmpd, basis::PIBasis,
                      cfg::AbstractConfiguration)
   evaluate_ed!(tmpd.A, tmpd.dA, tmpd.tmpd_basis1p, basis.basis1p, cfg)
   evaluate_ed!(AA, dAA, tmpd, basis, tmpd.A, tmpd.dA)
end

function _AA_local_adjoints!(dAAdA, A, iAA2iA, iAA, ord, _real)
   @assert length(dAAdA) >= ord
   # TODO - optimize?
   # Forward pass:
   @inbounds dAAdA[1] = 1
   @inbounds AAfwd = A[iAA2iA[iAA, 1]]
   @inbounds for a = 2:ord
      dAAdA[a] = AAfwd
      AAfwd *= A[iAA2iA[iAA, a]]
   end
   aa = _real(AAfwd)
   # backward pass
   @inbounds AAbwd = A[iAA2iA[iAA, ord]]
   @inbounds for a = ord-1:-1:1
      dAAdA[a] *= AAbwd
      AAbwd *= A[iAA2iA[iAA, a]]
   end

   return aa 
end

function evaluate_ed!(AA, dAA, tmpd, basis::PIBasis,
                      A::AbstractVector, dA::AbstractMatrix)
   dAAdA = tmpd.dAAdA
   orders = basis.spec.orders
   iAA2iA = basis.spec.iAA2iA

   for iAA = 1:length(basis)
      ord = orders[iAA]
      # ----- compute the local adjoints dAA / dA
      # dAAdA[a] ← ∏_{t ≂̸ a} A_{v_t}
      AA[iAA] = _AA_local_adjoints!(dAAdA, A, iAA2iA, iAA, orders[iAA], basis.real)

      # ----- now convert them into dAA / dX
      for j = 1:size(dA, 2)
         val = sum(dAAdA[a] * dA[iAA2iA[iAA, a], j] for a = 1:ord)
         dAA[iAA, j] = sum(dAAdA[a] * dA[iAA2iA[iAA, a], j]
                           for a = 1:ord) |> basis.real
      end
   end

end
