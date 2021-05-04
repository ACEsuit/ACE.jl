
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# modules external to our own eco-system, rigorously separate using and import

using Parameters: @with_kw

using Random: shuffle

import Base: ==, length

using LinearAlgebra: norm, dot, mul!, I

using StaticArrays

# -----------------------------------------------------------------------------
# JuLIP, ACE, etc : just use import throughout, this avoids bugs

import JuLIP

import JuLIP: alloc_temp, alloc_temp_d,
              cutoff,
              evaluate, evaluate_d,
              evaluate!, evaluate_d!,
              SitePotential,
              z2i, i2z, numz,
              read_dict, write_dict,
              AbstractCalculator,
              Atoms,
              chemical_symbol,
              fltype, rfltype

import JuLIP.MLIPs: IPBasis, alloc_B, alloc_dB, combine

import JuLIP.Potentials: ZList, SZList, zlist
import JuLIP: JVec, AtomicNumber
