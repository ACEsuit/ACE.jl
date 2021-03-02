# --------------------------------------------------------------------------
# ACE.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


export EuclideanVectorState,
       PositionState,
       SpeciesState,
       AtomState,
       ⊗

abstract type AbstractState end

abstract type ContinuousState <: AbstractState end

abstract type DiscreteState <: AbstractState end


abstract type AbstractSymmetry end

struct EuclideanO3Equivariance  <: AbstractSymmetry end

struct SphericalO3Equivariance  <: AbstractSymmetry end

struct O3Invariance <: AbstractSymmetry end

groupaction(X::AbstractState) = groupaction(typeof(X))


@doc raw"""
`struct EuclideanVectorState` : a $\mathbb{R}^3$-vector, which transforms under
the rotation group as
```math
g_Q \cdot {\bm r} = Q {\bm r}.
```
It typically defines a position or a force.
"""
struct EuclideanVectorState{T} <: ContinuousState
   rr::SVector{3, T}
   label::String
end

"same as EuclideanVectorState"
const PositionState{T} = EuclideanVectorState{T}

EuclideanVectorState(rr::SVector{3}) = EuclideanVectorState(rr, "𝒓")
EuclideanVectorState{T}(label::String = "𝒓") where {T} = EuclideanVectorState(zero(SVector{3, T}), label)
EuclideanVectorState(label::String = "𝒓") = EuclideanVectorState(zero(SVector{3, Float64}), label)

Base.length(X::EuclideanVectorState) = 3
groupaction(::Type{EuclideanVectorState}) = EuclideanO3Equivariance

Base.show(io::IO, s::EuclideanVectorState) = print(io, "$(s.label)$(s.rr)")

@doc raw"""
`struct SpeciesState` : a $\mathbb{Z}$ value, which is invariant under the
rotation group. It defines an atomic species.
"""
struct SpeciesState <: DiscreteState
   mu::AtomicNumber
   label::String
   SpeciesState(z_or_sym, label::String = "") =
         new(AtomicNumber(z_or_sym), label)
end

SpeciesState(label::String = "") = SpeciesState(AtomicNumber(0), label)

Base.length(X::SpeciesState) = 1
groupaction(::Type{SpeciesState}) = O3Invariance

Base.show(io::IO, s::SpeciesState) =
      print(io, "$(s.label)[$(chemical_symbol(s.mu))]")




# a starting point how to construct general states
# using a macro instead of writing them by hand
# macro state(name, args...)
#    @show name
#    for x in args
#       @assert x.args[1] === Symbol("=>")
#    end
#    fields = [:($(x.args[2])::$(x.args[3])) for x in args]
#    esc(quote struct $name <: AbstractState
#       $(fields...)
#       end
#    end)
# end


# An Example

"""
`struct AtomState` : basic implementation of the state of an Atom
consistent with original ACE.
"""
struct AtomState{T} <: AbstractState
   mu::AtomicNumber
   rr::SVector{3, T}
end

AtomState(mu, rr::AbstractVector{T}) where {T} =
   AtomState(AtomicNumber(mu), SVector{3, T}(rr...))
AtomState(T::Type) = AtomState(0, zero(SVector{3, T}))
AtomState(mu = 0, T::Type = Float64) = AtomState(mu, zero(SVector{3, T}))

Base.show(io::IO, X::AtomState) =
   print(io, ( SpeciesState(X.mu, "μ"),
               PositionState(X.rr, "𝒓") ))
