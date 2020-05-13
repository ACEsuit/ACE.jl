
module PairPotentials

import JuLIP
using JuLIP: JVec, JMat, Atoms, AtomicNumber
using JuLIP.MLIPs: IPBasis
using LinearAlgebra: norm, dot
using JuLIP.Potentials: ZList, SZList, @pot, @D,
                        PairPotential, SimplePairPotential
using StaticArrays: SMatrix

using SHIPs: ScalarBasis, _allfieldsequal

import JuLIP: evaluate!, evaluate_d!, cutoff,
              evaluate, evaluate_d,
              read_dict, write_dict,
              energy, forces, virial,
              alloc_temp, alloc_temp_d,
              z2i, i2z, numz


import JuLIP.MLIPs: alloc_B, alloc_dB

import Base: ==, length


include("pair_basis.jl")

include("pair_pot.jl")

include("repulsion.jl")

end
