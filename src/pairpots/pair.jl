
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


module PairPotentials

import JuLIP
using JuLIP: JVec, JMat, Atoms, AtomicNumber
using JuLIP.MLIPs: IPBasis
using LinearAlgebra: norm, dot
using JuLIP.Potentials: ZList, SZList, z2i, i2z, numz, @pot, PairPotential
using StaticArrays: SMatrix

using SHIPs: ScalarBasis

import JuLIP: evaluate!, evaluate_d!, cutoff,
              evaluate, evaluate_d,
              read_dict, write_dict,
              energy, forces, virial,
              alloc_temp, alloc_temp_d


import JuLIP.MLIPs: alloc_B, alloc_dB

import Base: ==, length

include("pair_basis.jl")

include("pair_pot.jl")

end
