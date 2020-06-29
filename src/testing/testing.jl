
module Testing

using LinearAlgebra: eigvals, eigen
import JuLIP.Potentials: F64fun
import JuLIP: Atoms, bulk, rattle!, positions, energy, forces, JVec,
              chemical_symbol, mat

include("../extimports.jl")
include("../shipimports.jl")

include("testmodel.jl")
include("testlsq.jl")


end
