
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------





# prototypes for space transforms and cutoffs
function transform end
function transform_d end
function fcut end
function fcut_d end

"""
`VecOrTup = Union{AbstractVector, Tuple}`
"""
const VecOrTup = Union{AbstractVector, Tuple}


abstract type ScalarBasis{T} <: IPBasis end

abstract type OneParticleBasis{T} <: IPBasis end

abstract type OnepBasisFcn end

# ------------------------------------------------------------
#  Abstract polynomial degree business

"""
`AbstractDegree` : object specifying a degree can be called via
`degree(D, arg)` or via `D(arg)`
"""
abstract type AbstractDegree end

(D::AbstractDegree)(args...) = degree(D, args...)

"""
`function degree(D::AbstractDegree, arg)` : compute some notion of degree of
the `arg` argument.
"""
function degree end

"""
interface functions for `OneParticleBasis`
"""
function add_into_A! end

"""
interface functions for `OneParticleBasis`
"""
function add_into_A_dA! end

"""
`function scaling(b, p)`:

a scaling factor for a basis functions ϕ, which gives a rought estimate on
   the magnitude of ∇ᵖϕ e.g.,
```
ϕ = r^n Ylm
```
has scaling factor `n^p + l^p`, though sharper estimates are also possible.
"""
function scaling end



using LinearAlgebra: Diagonal

diagonal_regulariser(basis; diff = 0) = Diagonal(scaling(basis, diff))

"""
every scalar basis must implement this
"""
function rand_radial end
