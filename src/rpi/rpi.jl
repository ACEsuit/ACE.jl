
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


include("rotations3d.jl")
using SHIPs.Rotations3D

# some basic degree types useful for RPI type constructions
# (this file also specifies the PSH1pBasisFcn
include("rpi_degrees.jl")

# the basic RPI type 1-particle basis
include("rpi_basic1pbasis.jl")

# RPI basis
include("rpi_basis.jl")

# RPI Potential
# include("ace_potential.jl")
