

# ----------- definitions of possible symmetries a 1-p basis may possess

# TODO: attach these properties to the one-particle basis sets

abstract type AbstractSymmetry end

struct EuclideanO3Equivariance  <: AbstractSymmetry end

struct SphericalO3Equivariance  <: AbstractSymmetry end

struct O3Invariance <: AbstractSymmetry end

groupaction(X::AbstractState) = groupaction(typeof(X))


# ----------- This file implements the abstract one-particle basis interface


function evaluate!(A, tmp, basis::OneParticleBasis,
                   cfg::AbstractConfiguration)
   fill!(A, 0)
   for X in cfg
      add_into_A!(A, tmp, basis, X)
   end
   return A
end

function evaluate!(A, tmp, basis::OneParticleBasis, X::AbstractState)
   fill!(A, 0)
   add_into_A!(A, tmp, basis, X)
   return A
end

evaluate_d(basis::OneParticleBasis, cfg::AbstractConfiguration) = 
   evaluate_ed!(alloc_B(basis, cfg), alloc_dB(basis, cfg), 
                  alloc_temp_d(basis, cfg), basis, cfg)


# TODO: first signs of significant rewrite need here: 
#       probably get rid of all non-allocating versions here...

function evaluate_ed!(A, dA, tmpd, basis::OneParticleBasis,
                      cfg::AbstractConfiguration)
   fill!(A, 0)
   # fill!(dA, zero(eltype(dA)))  # TODO: this should not be necessary!
   for (j, X) in enumerate(cfg)
      A[:] .+= evaluate(basis, X)
      dA[:, j] = evaluate_d(basis, X)
      # dAview = @view dA[:, j]      
      # add_into_A_dA!(A, dAview, tmpd, basis, X)
   end
   return dA
end


function evaluate_ed!(A, dA, tmpd, basis::OneParticleBasis, X::AbstractState)
   # fill!(A, 0)
   # add_into_A_dA!(A, dA, tmpd, basis, X)
   A[:] .= evaluate(basis, X) 
   dA[:] .= evaluate_d(basis, X)
   return dA
end

# -------------------- AD codes 

import Zygote, ChainRules
import Zygote: Buffer
import ChainRules: rrule, NoTangent, ZeroTangent

function evaluate(basis::OneParticleBasis, 
                  cfg::AbstractConfiguration)
   A = zeros(valtype(basis, cfg), length(basis))
   for X in cfg
      A[:] += evaluate(basis, X)
   end
   return A
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

function evaluate!(B, tmp, basis::One1pBasis, args...)
   B[1] = 1
   return B
end

function evaluate_d!(dB, tmp, basis::One1pBasis, args...)
   dB[1] = zero(eltype(dB))
   return dB
end

fltype(::One1pBasis) = Bool

symbols(::One1pBasis) = Symbol[]

indexrange(::One1pBasis) = NamedTuple()

isadmissible(b, ::One1pBasis) = true

degree(b, ::One1pBasis) = 0

get_spec(::One1pBasis) = [(1,)]
