
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------




# some basic degree types useful for ACE type constructions
# (this file also specifies the PSH1pBasisFcn
include("degrees.jl")

# the basic ACE type 1-particle basis
include("basic1pbasis.jl")




# # -----------------------------------
# # iterating over an m collection
# # -----------------------------------
#
# _mvec(::CartesianIndex{0}) = SVector(IntS(0))
#
# _mvec(mpre::CartesianIndex) = SVector(Tuple(mpre)..., - sum(Tuple(mpre)))
#
# struct MRange{N, TI, T2}
#    ll::SVector{N, TI}
#    cartrg::T2
# end
#
# Base.length(mr::MRange) = sum(_->1, _mrange(mr.ll))
#
# """
# Given an l-vector `ll` iterate over all combinations of `mm` vectors  of
# the same length such that `sum(mm) == 0`
# """
# _mrange(ll) = MRange(ll, Iterators.Stateful(
#                         CartesianIndices(ntuple(i -> -ll[i]:ll[i], length(ll)-1))))
#
# function Base.iterate(mr::MRange{1, TI}, args...) where {TI}
#    if isempty(mr.cartrg)
#       return nothing
#    end
#    while !isempty(mr.cartrg)
#       popfirst!(mr.cartrg)
#    end
#    return SVector{1, TI}(0), nothing
# end
#
# function Base.iterate(mr::MRange, args...)
#    while true
#       if isempty(mr.cartrg)
#          return nothing
#       end
#       mpre = popfirst!(mr.cartrg)
#       if abs(sum(mpre.I)) <= mr.ll[end]
#          return _mvec(mpre), nothing
#       end
#    end
#    error("we should never be here")
# end
