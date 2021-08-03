


# ----------- This file implements the abstract one-particle basis interface


function evaluate!(A, basis::OneParticleBasis, cfg::AbstractConfiguration)
   fill!(A, 0)
   for X in cfg
      add_into_A!(A, basis, X)
   end
   return A
end

function evaluate!(A, basis::OneParticleBasis, X::AbstractState)
   fill!(A, 0)
   add_into_A!(A, basis, X)
   return A
end

function evaluate_d!(dA, basis::OneParticleBasis, 
                     X::Union{AbstractState, AbstractConfiguration})
   A = acquire_B!(basis, X)
   evaluate_ed!(A, dA, basis, X)
   return dA 
end

function evaluate_ed!(A, dA, basis::OneParticleBasis,
                      cfg::AbstractConfiguration)
   fill!(A, 0)
   for (j, X) in enumerate(cfg)
      add_into_A_dA!(A, (@view dA[:, j]), basis, X)
   end
   return A, dA
end


function evaluate_ed!(A, dA, basis::OneParticleBasis, X::AbstractState)
   fill!(A, 0)
   add_into_A_dA!(A, dA, basis, X)
   return A, dA
end




# -------------------- AD codes 

import ChainRules
import ChainRules: rrule, NoTangent, ZeroTangent

function evaluate!(A, basis::OneParticleBasis, 
                      cfg::AbstractConfiguration)
   fill!(A, 0)
   for X in cfg
   add_into_A!(A, basis, X)
   end
   return A
end

function evaluate(basis::OneParticleBasis, 
                  cfg::AbstractConfiguration)
   # A = acquire!(ACE._pool, valtype(basis, cfg), (length(basis),))
   A = Vector{valtype(basis, cfg)}(undef, length(basis))
   return evaluate!(A, basis, cfg)
end



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
