
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------



module Descriptors

using JuLIP

using ACE: SparseSHIP, PolyTransform, PolyCutoff1s, SHIPBasis

export SHIPDescriptor, descriptors

"""
`SHIPDescriptor(; deg=nothing, wY=1.5, rcut=nothing, r0=1.0, p=2)`

This returns a `desc::SHIPBasis` object which can be interpreted as a descriptor
map. Call `descriptors(desc, py_at)` where `py_at` is a Python object
(of ase Atoms type) to get the descriptors.

Call `Dict(desc)` to obtain a dictionary that fully describes the descriptor
and can be serialised to a JSON file / deserialised.

* `bodyorder` : specify body-order; `bodyorder = 3` corresponds to SOAP??;
`bodyorder = 5` to SNAP??;
* `deg, wY` : specify polynomial degree restriction; the basis will contain all
tensor products `Pk * Ylm` such that `k + wY * l ≦ deg`.
* `rcut` : cutoff radius
* `r0` : an estimate for nearest-neighbour distance - not crucial
* `p` : specifies distance transform, u = (r/r0)^(-p); i.e., polynomials
`Pk` are polynomials in `u` not in `r`. (e.g. p = 1 => Coulomb coordinates)
"""
function SHIPDescriptor(species = :X;
                        bodyorder=3, deg=nothing,
                        rcut=nothing, r0=2.5, p=2,
                        kwargs... )
   spec = SparseSHIP(species, bodyorder-1, deg; kwargs...)
   trans = PolyTransform(p, r0)
   fcut = PolyCutoff1s(2, rcut)
   return SHIPBasis(spec, trans, fcut)
end


function descriptors(basis::SHIPBasis, _at)
   at = Atoms(_at)
   B = zeros(Float64, length(basis), length(at))
   for i = 1:length(at)
      B[:, i] = site_energy(basis, at, i)
   end
   return B
end

end
