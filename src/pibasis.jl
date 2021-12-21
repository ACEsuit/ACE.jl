
export PIBasis


# ---------------------- Implementation of the PIBasisSpec

"""
`struct PIBasisSpec`
"""
struct PIBasisSpec
   orders::Vector{Int}     # order (length) of ith basis function
   iAA2iA::Matrix{Int}     # where in A can we find the ith basis function
end

==(B1::PIBasisSpec, B2::PIBasisSpec) = _allfieldsequal(B1, B2)

Base.length(spec::PIBasisSpec) = length(spec.orders)

maxcorrorder(spec::PIBasisSpec) = size(spec.iAA2iA, 2)

function _get_pibfcn(spec0, Aspec, vv)
   vv1 = vv[2:end]
   vvnz = vv1[findall(vv1 .!= 0)]
   return (spec0[vv[1]], Aspec[vvnz])
end

function _get_pibfcn(Aspec, vv)
   vvnz = vv[findall(vv .!= 0)]
   return Aspec[vvnz]
end

# TODO: maybe instead of property == nothing, there should be a 
#       generic property with no symmetry attached to it. 

function PIBasisSpec( basis1p::OneParticleBasis,
                      symgrp::SymmetryGroup, 
                      Bsel::DownsetBasisSelector;
                      property = nothing,
                      filterfun = _->true,
                      init1pbasis = true )
   
   # we initialize the 1p-basis here; to prevent this it must be manually 
   # avoided by passing in init1pbasis = false 
   if init1pbasis
      init1pspec!(basis1p, Bsel)
   end

   # get the basis spec of the one-particle basis
   #  Aspec[i] described the basis function that will get written into A[i]
   Aspec = get_spec(basis1p)

   # we assume that `Aspec` is sorted by degree, but best to double-check this
   # since the notion of degree used to construct `Aspec` might be different
   # from the one used to construct AAspec.
   if !issorted(Aspec; by = b -> level(b, Bsel, basis1p))
      error("""PIBasisSpec : AAspec construction failed because Aspec is not
               sorted by degree. This could e.g. happen if an incompatible
               notion of degree was used to construct the 1-p basis spec.""")
   end
   # An AA basis function is given by a tuple 𝒗 = vv. Each index 𝒗ᵢ = vv[i]
   # corresponds to the basis function Aspec[𝒗ᵢ] and the tuple
   # 𝒗 = (𝒗₁, ...) to a product basis function
   #   ∏ A_{vₐ}
   tup2b = vv -> _get_pibfcn(Aspec, vv)

   #  degree or level of a basis function ↦ is it admissible?
   admissible = bb -> (level(bb, Bsel, basis1p) <= maxlevel(Bsel, basis1p))

   if property != nothing
      filter1 = bb -> filterfun(bb) && filter(bb, Bsel, basis1p) && filter(property, symgrp, bb)
   else
      filter1 = bb -> filterfun(bb) && filter(bb, Bsel, basis1p) 
   end


   # we can now construct the basis specification; the `ordered = true`
   # keyword signifies that this is a permutation-invariant basis
   maxord = maxorder(Bsel)
   AAspec = gensparse(; NU = maxorder(Bsel),
                        tup2b = tup2b,
                        admissible = admissible,
                        ordered = true,
                        maxvv = [length(Aspec) for _=1:maxord],
                        filter = filter1)

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
mutable struct PIBasis{BOP, REAL, TB, TA} <: ACEBasis
   basis1p::BOP             # a one-particle basis
   spec::PIBasisSpec
   real::REAL     # could be `real` or `identity` to keep AA complex
   # evaluator    # classic vs graph   
   B_pool::VectorPool{TB}
   dAA_pool::VectorPool{TA}
end

cutoff(basis::PIBasis) = cutoff(basis.basis1p)

==(B1::PIBasis, B2::PIBasis) = 
      ( (B1.basis1p == B2.basis1p) && 
        (B1.spec == B2.spec) && 
        (B1.real == B2.real) )

valtype(basis::PIBasis) = basis.real( valtype(basis.basis1p) )

valtype(basis::PIBasis, cfg::AbstractConfiguration) = 
      basis.real( valtype(basis.basis1p, cfg) )

gradtype(basis::PIBasis, cfgorX) = 
      basis.real( gradtype(basis.basis1p, cfgorX) )

Base.length(basis::PIBasis) = length(basis.spec)

# default symmetry group 
PIBasis(basis1p, Bsel::AbstractBasisSelector; kwargs...) = 
   PIBasis(basis1p, O3(), Bsel; kwargs...)

PIBasis(basis1p, symgrp, Bsel::AbstractBasisSelector; 
        isreal = false, kwargs...) =
   PIBasis(basis1p, 
           PIBasisSpec(basis1p, symgrp, Bsel; kwargs...),
           isreal ? Base.real : Base.identity )

function PIBasis(basis1p::OneParticleBasis, spec::PIBasisSpec, real)
   VT1 = valtype(basis1p)
   VT = real(VT1)  # default valtype 
   B_pool = VectorPool{VT}()
   dAA_pool = VectorPool{VT1}()
   return PIBasis(basis1p, spec, real, B_pool, dAA_pool)
end

get_spec(pibasis::PIBasis) =
   [ get_spec(pibasis, i) for i = 1:length(pibasis) ]

get_spec(pibasis::PIBasis, i::Integer) =
      get_spec.( Ref(pibasis.basis1p), get_spec(pibasis.spec, i) )

setreal(basis::PIBasis, isreal::Bool) =
   PIBasis(basis.basis1p, basis.spec, isreal)

maxcorrorder(basis::PIBasis) = maxcorrorder(basis.spec)

# TODO: this is a hack; cf. #68
function scaling(pibasis::PIBasis, p)
   _absvaluep(x::Number) = abs(x)^p
   _absvaluep(x::Symbol) = 0
   ww = zeros(Float64, length(pibasis))
   bspec = get_spec(pibasis)
   for i = 1:length(pibasis)
      for b in bspec[i]
         # TODO: revisit how this should be implemented for a general basis
         ww[i] += sum(_absvaluep, b)  #  abs.(values(b)).^p
      end
   end
   return ww
end


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



# -------------------------------------------------
# FIO codes

write_dict(basis::PIBasis) =
   Dict(  "__id__" => "ACE_PIBasis",
         "basis1p" => write_dict(basis.basis1p),
            "spec" => write_dict(basis.spec),
            "real" => basis.real == Base.real ? true : false )

read_dict(::Val{:ACE_PIBasis}, D::Dict) =
   PIBasis( read_dict(D["basis1p"]),
            read_dict(D["spec"]),
            D["real"] ? Base.real : Base.identity )

write_dict(spec::PIBasisSpec) =
   Dict( "__id__" => "ACE_PIBasisSpec",
         "orders" => spec.orders,
         "iAA2iA" => write_dict(spec.iAA2iA) )

read_dict(::Val{:ACE_PIBasisSpec}, D::Dict) =
   PIBasisSpec( D["orders"], read_dict(D["iAA2iA"]) )


# -------------------------------------------------
# Evaluation codes

function evaluate!(AA, basis::PIBasis, config::AbstractConfiguration)
   A = acquire_B!(basis.basis1p, config)   #  THIS ALLOCATES!!!! 
   evaluate!(A, basis.basis1p, config)
   fill!(AA, 1)
   for iAA = 1:length(basis)
      aa = one(eltype(A))
      for t = 1:basis.spec.orders[iAA]
         aa *= A[ basis.spec.iAA2iA[ iAA, t ] ]
      end
      AA[iAA] = basis.real(aa)
   end
   release_B!(basis.basis1p, A)
   return AA
end


# -------------------------------------------------
# gradients

function evaluate_ed!(AA, dAA, basis::PIBasis,
                      cfg::AbstractConfiguration, args...)  
   A = acquire_B!(basis.basis1p, cfg)
   dA = acquire_dB!(basis.basis1p, cfg)   # TODO: THIS WILL ALLOCATE!!!!!
   evaluate_ed!(A, dA, basis.basis1p, cfg, args...)
   evaluate_ed!(AA, dAA, basis, A, dA)
   release_dB!(basis.basis1p, dA)
   release_B!(basis.basis1p, A)
   return AA, dAA 
end

function _AA_local_adjoints!(dAAdA, A, iAA2iA, iAA, ord, _real)
   @assert length(dAAdA) >= ord
   # TODO - optimize a bit more? can move one operation out of the loop
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

_acquire_dAAdA!(basis::PIBasis) = acquire!(basis.dAA_pool, maxcorrorder(basis))

function evaluate_ed!(AA, dAA, basis::PIBasis,
                      A::AbstractVector, dA::AbstractMatrix)
   orders = basis.spec.orders
   iAA2iA = basis.spec.iAA2iA
   dAAdA = _acquire_dAAdA!(basis)

   # Must treat the constants separately. This is not so elegant and could 
   # maybe be improved? 
   if orders[1] == 0  # SHOULD BE THE ONLY ONE with ord=0!! 
      iAAinit = 2
      AA[1] = 1.0 
      dAA[1, :] .= Ref(zero(eltype(dAA)))
   else 
      iAAinit = 1
   end

   for iAA = iAAinit:length(basis)
      ord = orders[iAA]

      # ----- compute the local adjoints dAA / dA
      # dAAdA[a] ← ∏_{t ≂̸ a} A_{v_t}
      AA[iAA] = _AA_local_adjoints!(dAAdA, A, iAA2iA, iAA, orders[iAA], basis.real)

      # ----- now convert them into dAA / dX
      for j = 1:size(dA, 2)
         dAA[iAA, j] = sum(dAAdA[a] * dA[iAA2iA[iAA, a], j]
                           for a = 1:ord) |> basis.real
      end
   end

   return AA, dAA 
end
