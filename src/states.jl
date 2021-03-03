# --------------------------------------------------------------------------
# ACE.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


export EuclideanVectorState, DiscreteState

abstract type AbstractState end

abstract type AbstractContinuousState <: AbstractState end

abstract type AbstractDiscreteState <: AbstractState end

isdiscrete(::AbstractContinuousState) = false
isdiscrete(::AbstractDiscreteState) = true



@doc raw"""
`struct EuclideanVectorState` : a $\mathbb{R}^3$-vector, which transforms under
the rotation group as
```math
g_Q \cdot {\bm r} = Q {\bm r}.
```
It typically defines a position or a force.
"""
struct EuclideanVectorState{T} <: AbstractContinuousState
   rr::SVector{3, T}
   label::String
end

EuclideanVectorState(rr::SVector{3}) = EuclideanVectorState(rr, "𝒓")
EuclideanVectorState{T}(label::String = "𝒓") where {T} = EuclideanVectorState(zero(SVector{3, T}), label)
EuclideanVectorState(label::String = "𝒓") = EuclideanVectorState(zero(SVector{3, Float64}), label)

Base.length(X::EuclideanVectorState) = 3

Base.show(io::IO, s::EuclideanVectorState) = print(io, "$(s.label)$(s.rr)")

import Base: *
*(A::Union{Number, AbstractMatrix}, X::EuclideanVectorState) =
      EuclideanVectorState(A * X.rr, X.label)



@doc raw"""
`struct DiscreteState` : a state ``\mu`` specified by a discrete number of
possible values, e.g. ranging through ``\mathbb{Z}`` or ``\mathbb{Z}_p``.
Discrete states cannot possibly have a non-trivial transformation under
a continuoue group action, hence it is assumed to be invariant, i.e.,
```math
g_Q \cdot \mu = \mu
```
"""
struct DiscreteState{T, SYM} <: AbstractDiscreteState
   val::T
   _valsym::Val{SYM}
end

DiscreteState(sym::Symbol) = DiscreteState(0, Val(sym))
DiscreteState{T, SYM}(val::T) where {T, SYM} = DiscreteState(val, Val{SYM}())

Base.show(io::IO, s::DiscreteState{T, SYM}) where {T, SYM} =
         print(io, "$(SYM)[$(s.val)]")

Base.getproperty(s::DiscreteState{T, SYM}, sym) where {T, SYM} =
      sym == SYM ? getfield(s, :val) : getfield(s, sym)


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
