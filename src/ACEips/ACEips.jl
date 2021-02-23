
module ACEips

# import ACE: AbstractState, ContinousState, DiscreteState,
#             AtomicNumber

abstract type AbstractState end
abstract type ContinuousState end
abstract type DiscreteState end


struct EuclideanVectorState{T} <: ContinousState
   rr::SVector{3, T}
end



# -> specify the equivariance of this variable!
#    maybe incorporate this into the type information?

struct SpeciesState <: DiscreteState
   z::AtomicNumber
end





end
