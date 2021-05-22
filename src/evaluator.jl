


"""
`struct PIEvaluator` : specifies a PIEvaluator, which is basically defined
through a PIBasis and its coefficients
"""
mutable struct PIEvaluator{T, TPI} 
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

_get_eff_coeffs(basis::SymmetricBasis, c::AbstractVector{<: Number}) = 
      (transpose(c) * basis.A2B)[:]

set_params!(ev::PIEvaluator, basis::SymmetricBasis, coeffs::AbstractVector) = 
      set_params!(ev, basis.pibasis, _get_eff_coeffs(basis, coeffs))


PIEvaluator(basis::SymmetricBasis, c::AbstractVector{<: Number}) = 
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
alloc_temp(V::PIEvaluator{T}, maxN::Integer) where {T} =
   (
      tmp_pibasis = alloc_temp(V.pibasis, maxN),
   )



# compute one site energy
function evaluate!(tmp, V::PIEvaluator, cfg)
   A = evaluate!(tmp.tmp_pibasis.A, tmp.tmp_pibasis.tmp_basis1p,
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
   return Es
end


# # TODO: generalise the R, Z, allocation
# alloc_temp_d(::StandardEvaluator, V::PIEvaluator{T}, N::Integer) where {T} =
#       (
#       dAco = zeros(fltype(V.pibasis),
#                    maximum(length(V.pibasis.basis1p, iz) for iz=1:numz(V))),
#        tmpd_pibasis = alloc_temp_d(V.pibasis, N),
#        dV = zeros(JVec{real(T)}, N),
#         R = zeros(JVec{real(T)}, N),
#         Z = zeros(AtomicNumber, N)
#       )

# # compute one site energy
# function evaluate_d!(dEs, tmpd, V::PIEvaluator, ::StandardEvaluator,
#                      Rs::AbstractVector{<: JVec{T}},
#                      Zs::AbstractVector{AtomicNumber},
#                      z0::AtomicNumber
#                      ) where {T}
#    iz0 = z2i(V, z0)
#    basis1p = V.pibasis.basis1p
#    tmpd_1p = tmpd.tmpd_pibasis.tmpd_basis1p
#    Araw = tmpd.tmpd_pibasis.A

#    # stage 1: precompute all the A values
#    A = evaluate!(Araw, tmpd_1p, basis1p, Rs, Zs, z0)

#    # stage 2: compute the coefficients for the ∇A_{klm} = ∇ϕ_{klm}
#    dAco = tmpd.dAco
#    c = V.coeffs[iz0]
#    inner = V.pibasis.inner[iz0]
#    fill!(dAco, 0)
#    for iAA = 1:length(inner)
#       for α = 1:inner.orders[iAA]
#          CxA_α = c[iAA]
#          for β = 1:inner.orders[iAA]
#             if β != α
#                CxA_α *= A[inner.iAA2iA[iAA, β]]
#             end
#          end
#          iAα = inner.iAA2iA[iAA, α]
#          dAco[iAα] += CxA_α
#       end
#    end

#    # stage 3: get the gradients
#    fill!(dEs, zero(JVec{T}))
#    dAraw = tmpd.tmpd_pibasis.dA
#    for (iR, (R, Z)) in enumerate(zip(Rs, Zs))
#       evaluate_d!(Araw, dAraw, tmpd_1p, basis1p, R, Z, z0)
#       iz = z2i(basis1p, Z)
#       zinds = basis1p.Aindices[iz, iz0]
#       for iA = 1:length(basis1p, iz, iz0)
#          dEs[iR] += real(dAco[zinds[iA]] * dAraw[zinds[iA]])
#       end
#    end

#    return dEs
# end


