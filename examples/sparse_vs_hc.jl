
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------



using SHIPs

fcut = PolyCutoff1s(2, 5.0)
trans = PolyTransform(2, 2.5)

for deg in [8, 10, 12, 14, 16]
   spship = SHIPBasis(SparseSHIP(5, deg, 1.5), trans, fcut)
   hcship = SHIPBasis(HyperbolicCrossSHIP(5, deg, 1.5), trans, fcut)
   @show deg
   @show length(spship), length(hcship)
end
