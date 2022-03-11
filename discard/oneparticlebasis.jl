


_check_args_is_sym() = true
_check_args_is_sym(::Symbol) = true

# args... may be empty or a symbol  for partial derivatives
function evaluate_d!(dA, basis::OneParticleBasis, 
                     X::Union{AbstractState, AbstractConfiguration}, 
                     args...)
   @assert _check_args_is_sym(args...)
   A = acquire_B!(basis, X)
   evaluate_ed!(A, dA, basis, X, args...)
   release_B!(basis, A)
   return dA
end

# args... may be empty or a symbol  for partial derivatives
function evaluate_ed!(A, dA, basis::OneParticleBasis,
                      cfg::AbstractConfiguration, args...)
   @assert _check_args_is_sym(args...)
   fill!(A, 0)
   for (j, X) in enumerate(cfg)
      add_into_A_dA!(A, (@view dA[:, j]), basis, X, args...)
   end
   return A, dA
end

# args... may be empty or a symbol for partial derivatives
function evaluate_ed!(A, dA, basis::OneParticleBasis, X::AbstractState, args...)
   @assert _check_args_is_sym(args...)
   fill!(A, 0)
   add_into_A_dA!(A, dA, basis, X, args...)
   return A, dA
end


# fix evaluate_d and evaluate_ed for partial D, i.e. args = a symbol

evaluate_d(basis::ACEBasis, X::Union{AbstractState, AbstractConfiguration}, sym::Symbol) =  
      evaluate_d!( acquire_dB!(basis, X), basis, X, sym)

evaluate_ed(basis::ACEBasis, X::Union{AbstractState, AbstractConfiguration}, sym::Symbol) =  
      evaluate_ed!( acquire_B!(basis, X), acquire_dB!(basis, X), basis, X, sym )


function add_into_A_dA!(A, dAj, basis::OneParticleBasis, Xj, sym::Symbol)
   if sym in argsyms(basis)
      add_into_A_dA!(A, dAj, basis::OneParticleBasis, Xj)
   else 
      add_into_A!(A, basis, Xj)
      fill!(dAj, zero(eltype(dAj)))
   end
end

# if a basis hasn't implemented this ...
function add_into_A_dA!(A, dAj, basis::OneParticleBasis, Xj)
   ϕ = acquire_B!(basis, Xj)
   evaluate_ed!(ϕ, dAj, basis, Xj)
   @. A .+= ϕ
   release_B!(basis, ϕ)
   return nothing 
end

function add_into_A!(A, basis::OneParticleBasis, Xj)
   ϕ = acquire_B!(basis, Xj)
   evaluate!(ϕ, basis, Xj)
   @. A += ϕ
   release_B!(basis, ϕ)
   return nothing 
end


# -------------------- AD codes 

import ChainRules
import ChainRules: rrule, NoTangent, ZeroTangent


function _rrule_evaluate(basis::OneParticleBasis, 
                         cfg::ACEConfig, 
                         W )
   dXs = [ _rrule_evaluate(basis, X, W) for X in cfg ] 
   return DACEConfig(dXs)
end 

function rrule(::typeof(evaluate), basis::OneParticleBasis, 
                cfg::AbstractConfiguration) 
   return evaluate(basis, cfg), 
          w -> (NoTangent(), ZeroTangent(), 
                _rrule_evaluate(basis, cfg, w) )
end

# -------------------- END AD codes 



# function set_Aindices!(basis::OneParticleBasis)
#    NZ = numz(basis)
#    for iz0 = 1:NZ
#       idx = 0
#       for iz = 1:NZ
#          len = length(basis, iz, iz0)
#          basis.Aindices[iz, iz0] = (idx+1):(idx+len)
#          idx += len
#       end
#    end
#    return basis
# end
#
# get_basis_spec(basis::OneParticleBasis, iz0::Integer) =
#       get_basis_spec(basis, i2z(iz0))
#
# get_basis_spec(basis::OneParticleBasis, s::Symbol) =
#       get_basis_spec(basis, atomic_number(s))



# --------------------

"""
`struct One1p` : the 1-p basis where everything just maps to 1
"""
struct One1pBasis <: OneParticleBasis{Bool}
end

Base.length(::One1pBasis) = 1

function evaluate!(B, basis::One1pBasis, args...)
   B[1] = 1
   return B
end

function evaluate_d!(dB, basis::One1pBasis, args...)
   dB[1] = zero(eltype(dB))
   return dB
end

fltype(::One1pBasis) = Bool

symbols(::One1pBasis) = Symbol[]

indexrange(::One1pBasis) = NamedTuple()

isadmissible(b, ::One1pBasis) = true

degree(b, ::One1pBasis) = 0

get_spec(::One1pBasis) = [(1,)]
