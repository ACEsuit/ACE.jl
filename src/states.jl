
export EuclideanVectorState,
       PositionState,
       SpeciesState,
       generate_state,
       ⊗

abstract type AbstractState end

abstract type ContinuousState <: AbstractState end

abstract type DiscreteState <: AbstractState end


abstract type AbstractSymmetry end

struct EuclideanO3Equivariance  <: AbstractSymmetry end

struct SphericalO3Equivariance  <: AbstractSymmetry end

struct O3Invariance <: AbstractSymmetry end


groupaction(X::AbstractState) = groupaction(typeof(X))


struct EuclideanVectorState{T} <: ContinuousState
   rr::SVector{3, T}
   label::String
end

"same as EuclideanVectorState"
const PositionState{T} = EuclideanVectorState{T}

EuclideanVectorState(rr::SVector{3}) = EuclideanVectorState(rr, "")
EuclideanVectorState{T}(label::String = "") where {T} = EuclideanVectorState(zero(SVector{3, T}), label)
EuclideanVectorState(label::String = "") = EuclideanVectorState(zero(SVector{3, Float64}), label)

Base.length(X::EuclideanVectorState) = 3
groupaction(::Type{EuclideanVectorState}) = EuclideanO3Equivariance

Base.show(io::IO, s::EuclideanVectorState) = print(io, "$(s.label)$(s.rr)")


struct SpeciesState <: DiscreteState
   z::AtomicNumber
   label::String
end

SpeciesState(label::String = "") = SpeciesState(AtomicNumber(0), label)
SpeciesState(z_or_sym, label = "") = SpeciesState(AtomicNumber(z_or_sym), label)

Base.length(X::SpeciesState) = 1
groupaction(::Type{SpeciesState}) = O3Invariance

Base.show(io::IO, s::SpeciesState) = print(io, "$(s.label)$(s.z)")

# example

struct State{T <: Tuple} <: AbstractState
   vals::T
end

kron(s1::AbstractState, args...) =
   State( tuple( [ [s1]; [ arg for arg in args ] ]... ) )

kron(s1::Type{<: AbstractState}, args...) =
   State( tuple( [ [s1()]; [ arg() for arg in args ] ]... ) )

⊗(s1::AbstractState, s2::AbstractState) = kron(s1, s2)
⊗(s1::State{T}, s2::AbstractState) where {T} = kron(T..., s2)
⊗(s1::AbstractState, s2::State{T}) where {T} = s2 ⊗ s1
⊗(s1::State{T1}, s2::State{T2}) where {T1, T2} = kron(T1..., T2...)

Base.length(X::State) = length(X.vals)

Base.show(io::IO, s::State) = print(io, s.vals)
