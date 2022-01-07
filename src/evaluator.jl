


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
   release!(basis.B_pool, c̃)
   return ev 
end


_get_eff_coeffs!(c̃, basis::SymmetricBasis, c::AbstractVector) = 
   genmul!(c̃, transpose(basis.A2Bmap), c, *)


function _get_eff_coeffs(basis::SymmetricBasis, c::AbstractVector)
   # c̃ = acquire_B!(basis, size(basis.A2Bmap, 2))
   c̃ = _alloc_ctilde(basis,c)
   return _get_eff_coeffs!(c̃, basis, c) 
end


# TODO: we may need a second pool to allocate ctilde vectors...

# _acquire_ctilde(basis::SymmetricBasis, len_AA, c::AbstractVector{<: SVector}) = 
#    acquire!(basis.B_pool, len_AA, SVector{length(c[1]),eltype(basis.A2Bmap)})

# _acquire_ctilde(basis::SymmetricBasis, len_AA, c::AbstractVector{<: Number}) = 
#    acquire!(basis.B_pool, len_AA)

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


# _alloc_dAco(dAAdA, A, c̃::AbstractVector{<: SVector}) = 
#    zeros(SVector{length(c̃[1]),eltype(dAAdA)}, length(A))
   
# _alloc_dAco(dAAdA, A, c̃::AbstractVector{<: ACE.Invariant}) = 
#    zeros(ACE.Invariant{eltype(dAAdA)}, length(A))

# _alloc_dAco(dAAdA, A, c̃::AbstractVector{ACE.EuclideanVector{T}}) where {T} = 
#    zeros(ACE.EuclideanVector{promote_type(real(eltype(dAAdA)), T)}, length(A))

# _alloc_dAco(dAAdA, A, c̃::AbstractVector{<: SVector}) = 
#    zeros( promote_type(eltype(dAAdA), eltype(c̃)), length(A) )
   
# _alloc_dAco(dAAdA, A, c̃::AbstractVector{<: ACE.Invariant}) = 
#    zeros( promote_type(eltype(dAAdA), eltype(c̃)), length(A) ) 

# _alloc_dAco(dAAdA, A, c̃::AbstractVector{ACE.EuclideanVector{T}}) where {T} = ( 
#    @show eltype(dAAdA); 
#    @show eltype(c̃); 
#    @show promote_type(eltype(dAAdA), eltype(c̃)); 
#    zeros( promote_type(eltype(dAAdA), eltype(c̃)), length(A) ) )

_alloc_dAco(dAAdA::AbstractVector, A::AbstractVector, c̃) =
         zeros( promote_type(eltype(dAAdA), eltype(c̃)), length(A) )


# ------------------------------------------------------------
#   Standard Evaluation code
# ------------------------------------------------------------



evaluate(::LinearACEModel, V::ProductEvaluator, cfg::AbstractConfiguration) = 
      evaluate(V::ProductEvaluator, cfg)

# compute one "site energy"
function evaluate(V::ProductEvaluator, cfg::AbstractConfiguration)
   A = acquire_B!(V.pibasis.basis1p, cfg)
   evaluate!(A, V.pibasis.basis1p, cfg)
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
   release_B!(V.pibasis.basis1p, A)
   return val
end


grad_config!(g, m::LinearACEModel, V::ProductEvaluator, cfg::AbstractConfiguration) = 
      grad_config!(g, V, cfg)


# compute one site energy gradient 
function grad_config!(g, V::ProductEvaluator, cfg::AbstractConfiguration)
   basis1p = V.pibasis.basis1p
   pireal = V.pibasis.real 
   symreal = V.real
   A = acquire_B!(V.pibasis.basis1p, cfg)
   dA = acquire_dB!(V.pibasis.basis1p, cfg)
   dAAdA = _acquire_dAAdA!(V.pibasis)
   
   # stage 1: precompute all the A values
   evaluate_ed!(A, dA, basis1p, cfg)

   # stage 2: compute the coefficients for the ∇A_{nlm} = ∇ϕ_{nlm}
   # dAco[nlm] = coefficient of ∇A_{nlm} (via adjoints)
   c̃ = V.coeffs
   dAco =  _alloc_dAco(dAAdA, A, c̃) # tmpd.dAco  # TODO: ALLOCATION 
   spec = V.pibasis.spec

   if spec.orders[1] == 0; iAAinit = 2; else iAAinit = 1; end 

   fill!(dAco, zero(eltype(dAco)))
   @inbounds for iAA = iAAinit:length(spec)
      _AA_local_adjoints!(dAAdA, A, spec.iAA2iA, iAA, spec.orders[iAA], pireal)
      @fastmath for t = 1:spec.orders[iAA]
         dAco[spec.iAA2iA[iAA, t]] += dAAdA[t] * c̃[iAA] #trying to avoid using .* and complex.()
      end
   end
   
   # stage 3: get the gradients

   function _update_g!(iA, iX, ::AbstractProperty)
      g[iX] += symreal( coco_o_daa(dAco[iA], dA[iA, iX]) )
   end

   function _update_g!(iA, iX, c̃i::SVector)
      for iP = 1:length(c̃i)
         g[iX, iP] += symreal( coco_o_daa(dAco[iA][iP], dA[iA, iX]) )
      end
   end 

   fill!(g, zero(eltype(g)))
   for iX = 1:length(cfg), iA = 1:length(basis1p)
      _update_g!(iA, iX, c̃[1])
   end

   return g
end


function _rrule_evaluate(dp, model::LinearACEModel, cfg::AbstractConfiguration)
   g = acquire_grad_config!(model, cfg, zeros(eltype(model.c[1]), 3) )
   return _rrule_evaluate!(g, dp, model.evaluator, cfg)
end

# compute one site energy gradient 
function _rrule_evaluate!(g, dp, V::ProductEvaluator, cfg::AbstractConfiguration)

   _contract(x::AbstractVector, y::AbstractVector) = 
         sum( _contract(xi, yi) for (xi, yi) in zip(x, y) )

   _contract(x::Invariant, y::Number) = x * y

   basis1p = V.pibasis.basis1p
   _real = V.real
   A = acquire_B!(V.pibasis.basis1p, cfg)
   dA = acquire_dB!(V.pibasis.basis1p, cfg)
   dAAdA = _acquire_dAAdA!(V.pibasis)
   
   # stage 1: precompute all the A values
   evaluate_ed!(A, dA, basis1p, cfg)

   # stage 2: compute the coefficients for the ∇A_{klm} = ∇ϕ_{klm}
   c̃ = V.coeffs
   
   _rec_eltype(x::AbstractVector) = _rec_eltype(x[1])
   _rec_eltype(x::AbstractProperty) = typeof(x)

   dAco =  _alloc_dAco(dAAdA, A, zeros(_rec_eltype(c̃), 3)) # tmpd.dAco  # TODO: ALLOCATION 
   spec = V.pibasis.spec

   if spec.orders[1] == 0; iAAinit = 2; else iAAinit = 1; end 

   fill!(dAco, zero(eltype(dAco)))
   @inbounds for iAA = iAAinit:length(spec)
      _AA_local_adjoints!(dAAdA, A, spec.iAA2iA, iAA, spec.orders[iAA], _real)
      @fastmath for t = 1:spec.orders[iAA]
         dAco[spec.iAA2iA[iAA, t]] += dAAdA[t] * complex( _contract(c̃[iAA], dp) )
      end
   end

   # stage 3: get the gradients
   fill!(g, zero(eltype(g)))
   for iX = 1:length(cfg)
      for iA = 1:length(basis1p)
         g[iX] += _real(dAco[iA] * dA[iA, iX])
      end
   end

   return g
end



function adjoint_EVAL_D1(m::LinearACEModel, V::ProductEvaluator, cfg, w)
   basis1p = V.pibasis.basis1p
   dAAdA = zero(MVector{10, ComplexF64})   # TODO: VERY RISKY -> FIX THIS 
   A = zeros(ComplexF64, length(basis1p))
   TDX = gradtype(m.basis, cfg)
   dA = zeros(complex(TDX) , length(A), length(cfg))
   _real = V.real
   dAAw = acquire_B!(V.pibasis, cfg)
   dAw = similar(A)
   dB = zeros(Float64, length(m.c))   # TODO: fix hard-coded parameters!!!

   # [1] dA_t = ∑_j ∂ϕ_t / ∂X_j
   evaluate_ed!(A, dA, basis1p, cfg)
   fill!(dAw, 0)
   for k = 1:length(basis1p), j = 1:length(w)
      dAw[k] += _contract(w[j], dA[k, j])
   end

   # [2] dAA_k 
   spec = V.pibasis.spec
   fill!(dAAw, 0)
   if spec.orders[1] == 0; iAAinit=2; else; iAAinit=1; end 
   @inbounds for iAA = iAAinit:length(spec)
      _AA_local_adjoints!(dAAdA, A, spec.iAA2iA, iAA, spec.orders[iAA], _real)
      @fastmath for t = 1:spec.orders[iAA]
         vt = spec.iAA2iA[iAA, t]
         dAAw[iAA] += _real(dAw[vt] * dAAdA[t])
      end
   end

   genmul!(dB, m.basis.A2Bmap, dAAw, (a, x) -> a.val * x)

   release_B!(V.pibasis, dAAw)

   # [3] dB_k
   return dB
end



function adjoint_EVAL_D(m::LinearACEModel, V::ProductEvaluator, cfg, w)
   _contract = ACE.contract 

   basis1p = V.pibasis.basis1p
   dAAdA = zero(MVector{10, ComplexF64})   # TODO: VERY RISKY -> FIX THIS 
   A = zeros(ComplexF64, length(basis1p))
   TDX = gradtype(m.basis, cfg)
   dA = zeros(complex(TDX) , length(A), length(cfg))
   _real = V.real
   dAAw = acquire_B!(V.pibasis, cfg)
   dAw = similar(A)
   dB = similar(m.c)

   # [1] dA_t = ∑_j ∂ϕ_t / ∂X_j
   evaluate_ed!(A, dA, basis1p, cfg)
   fill!(dAw, 0)
   for k = 1:length(basis1p), j = 1:length(w)
      dAw[k] += _contract(w[j], dA[k, j])
   end

   # [2] dAA_k 
   spec = V.pibasis.spec
   fill!(dAAw, 0)
   if spec.orders[1] == 0; iAAinit=2; else; iAAinit=1; end 
   @inbounds for iAA = iAAinit:length(spec)
      _AA_local_adjoints!(dAAdA, A, spec.iAA2iA, iAA, spec.orders[iAA], _real)
      @fastmath for t = 1:spec.orders[iAA]
         vt = spec.iAA2iA[iAA, t]
         dAAw[iAA] += _real(dAw[vt] * dAAdA[t])
      end
   end

   genmul!(dB, m.basis.A2Bmap, dAAw, (a, x) -> a.val * x)

   release_B!(V.pibasis, dAAw)

   # [3] dB_k
   return dB
end

#for multiple properties. dispatch on the pullback input being a matrix. 
#Basically the same code, except for some parts where we loop over all properties. 
#We generate a list of size "nprop" and keep the same objects as for a single property 
#inside the list.
function adjoint_EVAL_D(m::LinearACEModel, V::ProductEvaluator, cfg, wt::Matrix)
   _contract = ACE.contract 
   
   basis1p = V.pibasis.basis1p
   dAAdA = zero(MVector{10, ComplexF64})   # TODO: VERY RISKY -> FIX THIS 
   A = zeros(ComplexF64, length(basis1p))
   TDX = gradtype(m.basis, cfg)
   dA = zeros(complex(TDX) , length(A), length(cfg))
   _real = V.real
   dAAw = [acquire_B!(V.pibasis, cfg) for _ in 1:length(m.c[1])]
   dAw = [similar(A) for _ in 1:length(m.c[1])]
   dB = similar(m.c)

   # [1] dA_t = ∑_j ∂ϕ_t / ∂X_j
   evaluate_ed!(A, dA, basis1p, cfg)
   for i in 1:length(m.c[1])
      fill!(dAw[i], 0)
   end
   for prop in 1:length(m.c[1])
      w = wt[:,prop]
      for k = 1:length(basis1p), j = 1:length(w)
         dAw[prop][k] += _contract(w[j], dA[k, j])
      end
   end

   # [2] dAA_k 
   spec = V.pibasis.spec
   for i in 1:length(m.c[1])
      fill!(dAAw[i], 0)
   end
   for prop in 1:length(m.c[1])
      if spec.orders[1] == 0; iAAinit=2; else; iAAinit=1; end 
      @inbounds for iAA = iAAinit:length(spec)
         _AA_local_adjoints!(dAAdA, A, spec.iAA2iA, iAA, spec.orders[iAA], _real)
         @fastmath for t = 1:spec.orders[iAA]
            vt = spec.iAA2iA[iAA, t]
            dAAw[prop][iAA] += _real(dAw[prop][vt] * dAAdA[t])
         end
      end
   end

   adjointgenmul!(dB, m.basis.A2Bmap, dAAw, (a, x) -> a.val * x)

   for i in 1:length(m.c[1])
      release_B!(V.pibasis, dAAw[i]) #TODO check that this indeed releases
   end

   # [3] dB_k
   return dB
end