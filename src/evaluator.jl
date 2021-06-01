


"""
`struct PIEvaluator` : specifies a PIEvaluator, which is basically defined
through a PIBasis and its coefficients
"""
mutable struct PIEvaluator{T, TPI <: PIBasis} 
   pibasis::TPI
   coeffs::Vector{T}
end


==(V1::PIEvaluator, V2::PIEvaluator) =
      (V1.pibasis == V2.pibasis) && (V1.coeffs == V2.coeffs)

# ------------------------------------------------------------
#   Initialisation and Parameter manipulation code
# ------------------------------------------------------------

function set_params!(ev::PIEvaluator, coeffs::AbstractVector)
   ev.coeffs[:] .= coeffs 
   return ev 
end 

function set_params!(ev::PIEvaluator, basis::PIBasis, coeffs::AbstractVector)
   @assert ev.pibasis === basis
   set_params!(ev, coeffs)
end 

_get_eff_coeffs!(ceff, basis::SymmetricBasis, c::AbstractVector) = 
      genmul!(ceff, transpose(basis.A2Bmap), c, *)

function _get_eff_coeffs(basis::SymmetricBasis, c::AbstractVector{<: Number}) 
   ceff = zeros(eltype(basis.A2Bmap), size(basis.A2Bmap, 2))
   return _get_eff_coeffs!(ceff, basis, c)
end

set_params!(ev::PIEvaluator, basis::SymmetricBasis, coeffs::AbstractVector) = 
      set_params!(ev, basis.pibasis, _get_eff_coeffs(basis, coeffs))


PIEvaluator(basis::SymmetricBasis, c::AbstractVector) = 
      PIEvaluator(basis.pibasis, _get_eff_coeffs(basis, c))



# ------------------------------------------------------------
#   FIO code
# ------------------------------------------------------------

write_dict(V::PIEvaluator) = Dict(
      "__id__" => "ACE_PIEvaluator",
     "pibasis" => write_dict(V.pibasis),
      "coeffs" => write_dict(V.coeffs) )

read_dict(::Val{:ACE_PIEvaluator}, D::Dict) =
   PIEvaluator( read_dict(D["pibasis"]),
                read_dict(D["coeffs"]) )


# ------------------------------------------------------------
#   Standard Evaluation code
# ------------------------------------------------------------


# TODO: generalise the R, Z, allocation
alloc_temp(V::PIEvaluator{T}, args...) where {T} =
   (
      tmp_pibasis = alloc_temp(V.pibasis),
   )


evaluate!(tmp, ::LinearACEModel, V::PIEvaluator, cfg::AbstractConfiguration) = 
      evaluate!(tmp, V::PIEvaluator, cfg)

# compute one site energy
function evaluate!(tmp, V::PIEvaluator, cfg::AbstractConfiguration)
   A = evaluate!(tmp.tmp_pibasis.A, tmp.tmp_pibasis.tmp1p,
                 V.pibasis.basis1p, cfg)
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
   return val
end



alloc_temp_d(m::LinearACEModel, V::PIEvaluator{T}, X::AbstractConfiguration, N::Integer = length(X)) where {T} =
      (
       dAco = zeros( complex(eltype(V.coeffs)),
                     length(V.pibasis.basis1p) ),
       tmpd_pibasis = alloc_temp_d(V.pibasis, X),
       dAAdA = (@MVector zeros( fltype(V.pibasis.basis1p),
                               maximum(V.pibasis.spec.orders) )),
      )

grad_config!(g, tmpd, ::LinearACEModel, V::PIEvaluator, cfg::AbstractConfiguration) = 
      evaluate_d!(g, tmpd, V::PIEvaluator, cfg)

# compute one site energy
function evaluate_d!(g, tmpd, V::PIEvaluator, cfg::AbstractConfiguration)
   basis1p = V.pibasis.basis1p
   tmpd_1p = tmpd.tmpd_pibasis.tmpd_basis1p
   A = tmpd.tmpd_pibasis.A
   dA = tmpd.tmpd_pibasis.dA
   _real = V.pibasis.real
   dAAdA = tmpd.dAAdA

   # stage 1: precompute all the A values
   evaluate_ed!(A, dA, tmpd_1p, basis1p, cfg)

   # stage 2: compute the coefficients for the ∇A_{klm} = ∇ϕ_{klm}
   dAco = tmpd.dAco
   c = V.coeffs
   spec = V.pibasis.spec
   fill!(dAco, zero(eltype(dAco)))
   @inbounds for iAA = 1:length(spec)
      _AA_local_adjoints!(dAAdA, A, spec.iAA2iA, iAA, spec.orders[iAA], _real)
      @fastmath for t = 1:spec.orders[iAA]
         dAco[spec.iAA2iA[iAA, t]] += dAAdA[t] * complex(c[iAA])
      end
   end

   # stage 3: get the gradients
   fill!(g, zero(eltype(g)))
   for iX = 1:length(cfg)
      @inbounds @fastmath for iA = 1:length(basis1p)
         g[iX] += _real.(dAco[iA] * dA[iA, iX])
      end
   end

   return g
end


function adjoint_EVAL_D(m::LinearACEModel, V::PIEvaluator, cfg, w) 
   basis1p = V.pibasis.basis1p
   tmpd_1p = alloc_temp_d(basis1p)
   dAAdA = zero(MVector{10, ComplexF64})
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
