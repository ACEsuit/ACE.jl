module SHIPs

using Reexport
@reexport using JuLIP

include("aux.jl")
include("prototypes.jl")
include("jacobi.jl")
include("sphericalharmonics.jl")
include("transforms.jl")
include("degrees.jl")
include("basis.jl")
include("fast.jl")
include("pair.jl")

end # module
