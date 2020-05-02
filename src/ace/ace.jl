
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


include("rotations3d.jl")

# some basic degree types useful for ACE type constructions
# (this file also specifies the PSH1pBasisFcn
include("degrees.jl")

# the basic ACE type 1-particle basis
include("basic1pbasis.jl")
