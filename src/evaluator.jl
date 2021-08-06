


"""
`struct ProductEvaluator` : specifies a ProductEvaluator, which is basically defined
through a PIBasis and its coefficients. The n-correlations are evaluated directly 
via a naive product of the atomic base.
"""
mutable struct ProductEvaluator{T, TPI <: PIBasis} 
   pibasis::TPI        # AA basis from ACE papers
   coeffs::Vector{T}   # c̃ coefficients from ACE papers 
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
      ProductEvaluator(basis.pibasis, _get_eff_coeffs(basis, c))


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
   c̃ = acquire!(basis.B_pool, len_AA)
   _get_eff_coeffs!(c̃, basis, c)
   set_params!(ev, basis.pibasis, c̃)
   release!(basis.B_pool, c̃)
   return ev 
end


_get_eff_coeffs!(c̃, basis::SymmetricBasis, c::AbstractVector) = 
      genmul!(c̃, transpose(basis.A2Bmap), c, *)

function _get_eff_coeffs(basis::SymmetricBasis, c::AbstractVector{<: Number})
   # c̃ = acquire_B!(basis, size(basis.A2Bmap, 2))
   c̃ = zeros(eltype(basis.A2Bmap), size(basis.A2Bmap, 2))
   return _get_eff_coeffs!(c̃, basis, c) 
end




# ------------------------------------------------------------
#   Standard Evaluation code
# ------------------------------------------------------------



evaluate(::LinearACEModel, V::ProductEvaluator, cfg::AbstractConfiguration) = 
      evaluate(V::ProductEvaluator, cfg)

# compute one "site energy"
function evaluate(V::ProductEvaluator, cfg::AbstractConfiguration)
   A = acquire_B!(V.pibasis.basis1p, cfg)
   A = evaluate!(A, V.pibasis.basis1p, cfg)
   spec = V.pibasis.spec
   _real = V.pibasis.real
   # initialize output with a sensible type 
   val = zero(eltype(V.coeffs)) * _real(zero(eltype(A)))  
   @inbounds for iAA = 1:length(spec)
      aa = A[spec.iAA2iA[iAA, 1]]
      for t = 2:spec.orders[iAA]
         aa *= A[spec.iAA2iA[iAA, t]]
      end
      val += _real(aa) * V.coeffs[iAA]
   end
   release_B!(V.pibasis.basis1p, cfg)
   return val
end


grad_config!(g, m::LinearACEModel, V::ProductEvaluator, cfg::AbstractConfiguration) = 
      grad_config!(g, V, cfg)

# compute one site energy
function grad_config!(g, V::ProductEvaluator, cfg::AbstractConfiguration)
   basis1p = V.pibasis.basis1p
   _real = V.pibasis.real
   A = acquire_B!(V.pibasis.basis1p, cfg)
   dA = acquire_dB!(V.pibasis.basis1p, cfg)
   dAAdA = _acquire_dAAdA!(V.pibasis)

   # stage 1: precompute all the A values
   evaluate_ed!(A, dA, basis1p, cfg)

   # stage 2: compute the coefficients for the ∇A_{klm} = ∇ϕ_{klm}
   dAco = zeros(eltype(dAAdA), length(A)) # tmpd.dAco  # TODO: ALLOCATION 
   c̃ = V.coeffs
   spec = V.pibasis.spec
   fill!(dAco, zero(eltype(dAco)))
   @inbounds for iAA = 1:length(spec)
      _AA_local_adjoints!(dAAdA, A, spec.iAA2iA, iAA, spec.orders[iAA], _real)
      @fastmath for t = 1:spec.orders[iAA]
         dAco[spec.iAA2iA[iAA, t]] += dAAdA[t] * complex(c̃[iAA])
      end
   end

   # stage 3: get the gradients
   fill!(g, zero(eltype(g)))
   for iX = 1:length(cfg)
      @inbounds @fastmath for iA = 1:length(basis1p)
         g[iX] += _real(dAco[iA] * dA[iA, iX])
      end
   end

   return g
end


function adjoint_EVAL_D(m::LinearACEModel, V::ProductEvaluator, cfg, w) 
   basis1p = V.pibasis.basis1p
   tmpd_1p = alloc_temp_d(basis1p)
   dAAdA = zero(MVector{10, ComplexF64})   # TODO: VERY RISKY -> FIX THIS 
   A = zeros(ComplexF64, length(basis1p))
   dA = zeros(SVector{3, ComplexF64}, length(A), length(cfg))
   _real = V.pibasis.real
   dAAw = alloc_B(V.pibasis)
   dAw = similar(A)
   dB = similar(m.c)

   # [1] dA_t = ∑_j ∂ϕ_t / ∂X_j
   evaluate_ed!(A, dA, tmpd_1p, basis1p, cfg)
   fill!(dAw, 0)
   for k = 1:length(basis1p), j = 1:length(w)
      dAw[k] += sum(dA[k, j] .* w[j])
   end

   # [2] dAA_k 
   spec = V.pibasis.spec
   fill!(dAAw, 0)
   @inbounds for iAA = 1:length(spec)
      _AA_local_adjoints!(dAAdA, A, spec.iAA2iA, iAA, spec.orders[iAA], _real)
      @fastmath for t = 1:spec.orders[iAA]
         vt = spec.iAA2iA[iAA, t]
         dAAw[iAA] += _real(dAw[vt] * dAAdA[t])
      end
   end

   genmul!(dB, m.basis.A2Bmap, dAAw, (a, x) -> a.val * x)

   # [3] dB_k
   return dB
end
