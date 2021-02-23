
export EuclideanVectorState,
       PositionState,
       SpeciesState,
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



"""
`struct State` : Cartesian product state. If `s1, ..., sn` are `AbstractState`s
then `s1 ⊗ ... ⊗ sn` will be of type `State` and defines the Cartesian product.
This operation goes hand-in-hand with the tensor product operation on
the corresponding one-particle bases.

Remarks:
(1) The symbol `⊗` is used instead of `×` which in Julia is reserved for the
cross product.
(2) Not clear this `State` struct is a good idea, it seems that the way we
access the fields has very poor performance. Probably best to just
do this by hand.
"""
struct State{T <: NamedTuple} <: AbstractState
   __vals::T
end

Base.getproperty(X::State, sym::Symbol) = (
      sym === :__vals ? getfield(X, :__vals)
                      : getfield(getfield(X.__vals, sym), sym) )

function kron(s1::AbstractState, args...)
   states = tuple(s1, args...)
   names = getindex.(fieldnames.(typeof.(states)), 1)
   return State( NamedTuple{names}(states) )
end

kron(s1::Type{<: AbstractState}, args...) =
   State( tuple( [ [s1()]; [ arg() for arg in args ] ]... ) )

⊗(s1::AbstractState, s2::AbstractState) = kron(s1, s2)
⊗(s1::State{T}, s2::AbstractState) where {T} = kron(T..., s2)
⊗(s1::AbstractState, s2::State{T}) where {T} = s2 ⊗ s1
⊗(s1::State{T1}, s2::State{T2}) where {T1, T2} = kron(T1..., T2...)

Base.length(X::State) = length(X.vals)

Base.show(io::IO, X::State) = print(io, X.__vals)


# a starting point how to construct general states
# macro state(name, args...)
#    @show name
#    for x in args
#       @assert x.args[1] === Symbol("=>")
#    end
#    fields = [:($(x.args[2])::$(x.args[3])) for x in args]
#    esc(quote struct $name
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
