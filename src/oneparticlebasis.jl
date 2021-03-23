
# --------------------------------------------------------------------------
# ACE.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


# ----------- definitions of possible symmetries a 1-p basis may possess

# TODO: attach these properties to the one-particle basis sets

abstract type AbstractSymmetry end

struct EuclideanO3Equivariance  <: AbstractSymmetry end

struct SphericalO3Equivariance  <: AbstractSymmetry end

struct O3Invariance <: AbstractSymmetry end

groupaction(X::AbstractState) = groupaction(typeof(X))


# ----------- This file implements the abstract one-particle basis interface

function alloc_B(basis::OneParticleBasis, args...)
   T = fltype(basis)
   return zeros(T, length(basis))
end

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





# function evaluate_d!(A, dA, tmpd, basis::OneParticleBasis,
#                      Rs, Zs::AbstractVector, z0)
#    fill!(A, 0)
#    fill!(dA, zero(eltype(dA)))  # TODO: this should not be necessary!
#    iz0 = z2i(basis, z0)
#    for (j, (R, Z)) in enumerate(zip(Rs, Zs))
#       iz = z2i(basis, Z)
#       Aview = @view A[basis.Aindices[iz, iz0]]
#       dAview = @view dA[basis.Aindices[iz, iz0], j]
#       add_into_A_dA!(Aview, dAview, tmpd, basis, R, iz, iz0)
#    end
#    return dA
# end
#
#
#
# function evaluate_d!(A, dA, tmpd, basis::OneParticleBasis,
#                      R, z::AtomicNumber, z0)
#    fill!(A, 0)
#    iz, iz0 = z2i(basis, z), z2i(basis, z0)
#    Aview = @view A[basis.Aindices[iz, iz0]]
#    dAview = @view dA[basis.Aindices[iz, iz0]]
#    add_into_A_dA!(Aview, dAview, tmpd, basis, R, iz, iz0)
#    return dA
# end
#




# # TODO: fix the dispatch structure on alloc_dB
# alloc_dB(basis::OneParticleBasis, ::JVec) = alloc_dB(basis, 1)
# alloc_dB(basis::OneParticleBasis, Rs::AbstractVector{<: JVec}, args...) =
#       alloc_dB(basis, length(Rs))
# alloc_dB(basis::OneParticleBasis) = alloc_dB(basis, 1)
#
# function alloc_dB(basis::OneParticleBasis, maxN::Integer)
#    NZ = numz(basis)
#    maxlen = maximum( sum( length(basis, iz, iz0) for iz = 1:NZ )
#                      for iz0 = 1:NZ )
#    T = fltype(basis)
#    return zeros(JVec{T}, (maxlen, maxN))
# end



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

evaluate!(B, tmp, basis::One1pBasis, Xj, Xi) = (B[1] = 1; B)

fltype(::One1pBasis) = Bool

symbols(::One1pBasis) = Symbol[]

indexrange(::One1pBasis) = Dict()

isadmissible(b, ::One1pBasis) = true

degree(b, ::One1pBasis) = 0

get_spec(::One1pBasis) = [(1,)]
