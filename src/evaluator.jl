


"""
`struct ProductEvaluator` : specifies a ProductEvaluator, which is basically defined
through a PIBasis and its coefficients. The n-correlations are evaluated directly 
via a naive product of the atomic base.
"""
mutable struct ProductEvaluator{T, TPI <: PIBasis, REAL}
   pibasis::TPI        # AA basis from ACE papers
   coeffs::Vector{T}   # c̃ coefficients from ACE papers 
   real::REAL          # the real operation stored in the SymmetricBasis
end


==(V1::ProductEvaluator, V2::ProductEvaluator) =
      (V1.pibasis == V2.pibasis) && (V1.coeffs == V2.coeffs)


# ----------- FIO 

write_dict(ev::ProductEvaluator) = Dict( "__id__" => "ACE_ProductEvaluator" )

read_dict(::Val{:ACE_ProductEvaluator}, D::Dict, basis, c) = ProductEvaluator(basis, c)

# ------------------------------------------------------------
#   Initialisation and Parameter manipulation code
# ------------------------------------------------------------

ProductEvaluator(basis::SymmetricBasis, c::AbstractVector) = 
      ProductEvaluator(basis.pibasis, _get_eff_coeffs(basis, c), basis.real)


# basic setter interface without any checks 
function set_params!(ev::ProductEvaluator, c̃::AbstractVector)
   ev.coeffs[:] .= c̃ 
   return ev 
end 

# trivial setter when the parameters already come in the c̃ format (AA-basis)
function set_params!(ev::ProductEvaluator, basis::PIBasis, c̃::AbstractVector)
   @assert ev.pibasis === basis
   set_params!(ev, c̃)
end

# if parameters come in c (B-basis) format then they first need to be converted 
# to c̃ format (AA basis)
function set_params!(ev::ProductEvaluator, basis::SymmetricBasis, c::AbstractVector)
   len_AA = length(ev.pibasis)
   @assert len_AA == size(basis.A2Bmap, 2)
   c̃ = _acquire_ctilde(basis, len_AA, c)
   _get_eff_coeffs!(c̃, basis, c)
   set_params!(ev, basis.pibasis, c̃)
   release!(c̃)
   return ev 
end


_get_eff_coeffs!(c̃, basis::SymmetricBasis, c::AbstractVector) = 
   genmul!(c̃, transpose(basis.A2Bmap), c, *)


function _get_eff_coeffs(basis::SymmetricBasis, c::AbstractVector)
   c̃ = _alloc_ctilde(basis,c)
   return _get_eff_coeffs!(c̃, basis, c) 
end


# TODO: we may need a second pool to allocate ctilde vectors...

struct _One <: Number 
end

import Base: * 
*(x, ::_One) = x
*(::_One, x) = x
*(x::AbstractProperty, ::_One) = x
*(::_One, x::AbstractProperty) = x
*(x::StaticArray, ::_One) = x
*(::_One, x::StaticArray) = x
*(::ACE._One, x::ACE.XState) = x
*(x::ACE.XState, ::ACE._One) = x

_acquire_ctilde(basis::SymmetricBasis, len_AA, c::AbstractVector{<: Number}) = 
      zeros(promote_type(eltype(basis.A2Bmap), eltype(c)), len_AA)

_acquire_ctilde(basis::SymmetricBasis, len_AA, c::AbstractVector{<: SVector}) = 
      zeros(SVector{length(c[1]), 
                    promote_type(eltype(basis.A2Bmap), eltype(c[1]))
                   } , len_AA )

_alloc_ctilde(basis::SymmetricBasis, c::AbstractVector{<: SVector}) = 
      zeros(SVector{length(c[1]),eltype(basis.A2Bmap)}, size(basis.A2Bmap, 2))
   
_alloc_ctilde(basis::SymmetricBasis, c::AbstractVector{<: Number}) = 
      zeros(eltype(basis.A2Bmap), size(basis.A2Bmap, 2))

_alloc_dAco(dAAdA::AbstractVector, A::AbstractVector, c̃::AbstractArray, args...) =
            _alloc_dAco(dAAdA, A, c̃[1], args...)

function _alloc_dAco(dAAdA::AbstractVector, A::AbstractVector, 
                     c̃::Union{TP, SVector{N, TP}}, dp = _One()
                     ) where {N, TP <: AbstractProperty} 
   c̃_dp = contract(c̃, dp)   
   @show eltype(dAAdA)
   @show typeof(c̃_dp)
   @show promote_type(eltype(dAAdA), eltype(c̃_dp))
   zeros( promote_type(eltype(dAAdA), typeof(c̃_dp)), length(A) )
end

# ------------------------------------------------------------
#   Standard Evaluation code
# ------------------------------------------------------------



evaluate(::LinearACEModel, V::ProductEvaluator, cfg::AbstractConfiguration) = 
      evaluate(V::ProductEvaluator, cfg)

# compute one "site energy"
function evaluate(V::ProductEvaluator, cfg::AbstractConfiguration)
   A = evaluate(V.pibasis.basis1p, cfg)
   spec = V.pibasis.spec
   pireal = V.pibasis.real 
   symreal = V.real
   # initialize output with a sensible type 
   val = symreal(zero(eltype(V.coeffs)) * pireal(zero(eltype(A))))

   # constant (0-order)
   if spec.orders[1] == 0 
      val += V.coeffs[1]
      iAAinit = 2 
   else 
      iAAinit = 1
   end

   @inbounds for iAA = iAAinit:length(spec)
      aa = A[spec.iAA2iA[iAA, 1]]
      for t = 2:spec.orders[iAA]
         aa *= A[spec.iAA2iA[iAA, t]]
      end
      val += symreal(pireal(aa) * V.coeffs[iAA])
   end

   release!(A)
   return val
end


grad_config(m::LinearACEModel, V::ProductEvaluator, cfg::UConfig) = 
      _rrule_evaluate(_One(), m, V, cfg)


_rrule_evaluate(dp, m::LinearACEModel, cfg::UConfig) = 
       _rrule_evaluate(dp, m, m.evaluator, cfg)


# NB - testing shows that pre-allocating everything gains about 10% for small 
#      configs and ca 20% for larger, more realistic configs. 
#      worth doing at some point, but not really an immediate priority!
function _rrule_evaluate(dp, m::LinearACEModel, V::ProductEvaluator, cfg::UConfig)
   basis1p = V.pibasis.basis1p
   pireal = V.pibasis.real 
   symreal = V.real
   c̃ = V.coeffs
   spec = V.pibasis.spec
   
   # stage 1: precompute all the A values
   A, dA = evaluate_ed(basis1p, cfg)

   # stage 2: compute the coefficients for the ∇A_{nlm} = ∇ϕ_{nlm}
   # dAco[nlm] = coefficient of ∇A_{nlm} (via adjoints)

   if spec.orders[1] == 0; iAAinit = 2; else iAAinit = 1; end 

   dAAdA = _acquire_dAAdA!(V.pibasis, A)
   T_dAco = typeof(dAAdA[1] * contract(dp, c̃[1]))
   dAco = zeros(T_dAco, length(A))

   @inbounds for iAA = iAAinit:length(spec)
      _AA_local_adjoints!(dAAdA, A, spec.iAA2iA, iAA, spec.orders[iAA], pireal)
      @fastmath for t = 1:spec.orders[iAA]
         dAco[spec.iAA2iA[iAA, t]] += dAAdA[t] * contract(dp, c̃[iAA])
      end
   end
   
   # stage 3: get the gradients
   GT = typeof(symreal( coco_o_daa(dAco[1], dA[1,1]) )) # promote_type(eltype(dA), eltype(dAAdA), eltype(A)))
   g = zeros(GT, length(cfg))
   for iX = 1:length(cfg), iA = 1:length(basis1p)
      g[iX] += symreal( coco_o_daa(dAco[iA], dA[iA, iX]) )
   end

   release!(A)
   release!(dA) 
   release!(dAco)
   release!(dAAdA)

   return g
end



function adjoint_EVAL_D(m::LinearACEModel, V::ProductEvaluator, cfg, w)
   basis1p = V.pibasis.basis1p
   _real = V.real

   # [1] dA_t = ∑_j ∂ϕ_t / ∂X_j
   A, dA = evaluate_ed(V.pibasis.basis1p, cfg)

   # some allocations based on the types of A, dA 
   dAAdA = _acquire_dAAdA!(V.pibasis, A)

   _dAw = contract(w[1], dA[1])
   T_dAw = typeof(_dAw)
   dAw = zeros(T_dAw, length(V.pibasis.basis1p))

   T_dAAw = typeof(_real(_dAw * dAAdA[1]))
   dAAw = zeros(T_dAAw, length(V.pibasis))

   for k = 1:length(basis1p), j = 1:length(w)
      dAw[k] += contract(w[j], dA[k, j])
   end

   # [2] dAA_k 
   spec = V.pibasis.spec
   if spec.orders[1] == 0; iAAinit=2; else; iAAinit=1; end 
   @inbounds for iAA = iAAinit:length(spec)
      ord = spec.orders[iAA]
      _AA_local_adjoints!(dAAdA, A, spec.iAA2iA, iAA, ord, _real)
      @fastmath for t = 1:ord
         vt = spec.iAA2iA[iAA, t]
         dAAw[iAA] += _real(dAw[vt] * dAAdA[t])
      end
   end

   dB = m.basis.A2Bmap * dAAw

   release!(A)
   release!(dA)   

   # [3] dB_k
   return dB
end



# #for multiple properties. dispatch on the pullback input being a matrix. 
# #Basically the same code, except for some parts where we loop over all properties. 
# #We generate a list of size "nprop" and keep the same objects as for a single property 
# #inside the list.
# function adjoint_EVAL_D(m::LinearACEModel, V::ProductEvaluator, cfg, wt::Matrix)
#    _contract = ACE.contract 
   
#    basis1p = V.pibasis.basis1p
#    dAAdA = zero(MVector{10, ComplexF64})   # TODO: VERY RISKY -> FIX THIS 
#    A = zeros(ComplexF64, length(basis1p))
#    TDX = gradtype(m.basis, cfg)
#    dA = zeros(complex(TDX) , length(A), length(cfg))   
#    _real = V.real
#    dAAw = [acquire_B!(V.pibasis, cfg) for _ in 1:length(m.c[1])]
#    dAw = [similar(A) for _ in 1:length(m.c[1])]
#    dB = similar(m.c)

#    # [1] dA_t = ∑_j ∂ϕ_t / ∂X_j
#    evaluate_ed!(A, dA, basis1p, cfg)
#    for i in 1:length(m.c[1])
#       fill!(dAw[i], 0)
#    end
#    for prop in 1:length(m.c[1])
#       w = wt[:,prop]
#       for k = 1:length(basis1p), j = 1:length(w)
#          dAw[prop][k] += _contract(w[j], dA[k, j])
#       end
#    end

#    # [2] dAA_k 
#    spec = V.pibasis.spec
#    for i in 1:length(m.c[1])
#       fill!(dAAw[i], 0)
#    end
#    for prop in 1:length(m.c[1])
#       if spec.orders[1] == 0; iAAinit=2; else; iAAinit=1; end 
#       @inbounds for iAA = iAAinit:length(spec)
#          _AA_local_adjoints!(dAAdA, A, spec.iAA2iA, iAA, spec.orders[iAA], _real)
#          @fastmath for t = 1:spec.orders[iAA]
#             vt = spec.iAA2iA[iAA, t]
#             dAAw[prop][iAA] += _real(dAw[prop][vt] * dAAdA[t])
#          end
#       end
#    end

#    adjointgenmul!(dB, m.basis.A2Bmap, dAAw, (a, x) -> a.val * x)

#    for i in 1:length(m.c[1])
#       release_B!(V.pibasis, dAAw[i]) #TODO check that this indeed releases
#    end

#    # [3] dB_k
#    return dB
# end