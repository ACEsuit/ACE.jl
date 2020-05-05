
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
