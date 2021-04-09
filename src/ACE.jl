
module ACE

using Reexport

# external imports that are useful for all submodules
include("extimports.jl")
include("baseimports.jl")

include("auxiliary.jl")

include("rotations3d.jl")
using ACE.Rotations3D

include("states.jl")
include("properties.jl")


include("prototypes.jl")


# basic polynomial building blocks
include("polynomials/sphericalharmonics.jl")
include("polynomials/transforms.jl"); @reexport using ACE.Transforms
include("polynomials/orthpolys.jl"); @reexport using ACE.OrthPolys

# The One-particle basis is the first proper building block
include("oneparticlebasis.jl")

# three specific 1p-bases that are useful
# TODO: species basis should be moved into the atomistic modelling toolkit
include("Ylm1pbasis.jl")
include("Rn1pbasis.jl")

include("product_1pbasis.jl")

include("sparsegrids.jl")

# TODO -> move elsewhere!!!
# include("rpi/rpi_degrees.jl")

# leave this for much later ...
# include("grapheval.jl")

# the permutation-invariant basis: this is a key building block
# for other bases but can also be a useful export itself
include("pibasis.jl")

include("symmbasis.jl")

# include("pipot.jl")


# # orthogonal basis
# include("orth.jl")
#
# lots of stuff related to random samples:
#  - random configurations
#  - random potentials
#  ...
include("random.jl")
@reexport using ACE.Random


include("utils.jl")
@reexport using ACE.Utils

# include("utils/importv5.jl")
# include("compat/compat.jl")
# include("export/export.jl")


include("testing/testing.jl")
include("testing/wigner.jl")

# - bond model
# - pure basis
# - real basis
# - regularisers
# - descriptors
# - random potentials


end # module
