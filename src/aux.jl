
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------



# -----------------------------------
# iterating over an m collection
# -----------------------------------

_mvec(::CartesianIndex{0}) = SVector(IntS(0))

_mvec(mpre::CartesianIndex) = SVector(Tuple(mpre)..., - sum(Tuple(mpre)))

"""
Given an l-vector `ll` iterate over all combinations of `mm` vectors  of
the same length such that `sum(mm) == 0`
"""
_mrange(ll) = (
                  _mvec(mpre) for mpre in CartesianIndices(
                                ntuple(i -> -ll[i]:ll[i], length(ll)-1) )
              )
