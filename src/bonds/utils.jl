
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


using PoSH: PolyTransform, IdTransform
# ------ Basis generation

function envpairbasis(species, bo, deg0, rcut0, degenv;
            rnn = species == :X ? 2.7 : JuLIP.rnn(species),
            rcutenvfact = 1.9,
            rinenvfact =
            rcutr = rcutenvfact * rnn,
            rinr = 0.66 * rnn,
            rcutz = rcutenvfact * rnn + 0.5 * rcut0,
            wenv = 4.0, wr = 1.0, wz = 1.0, wθ = 1.0,
            r0trans = PolyTransform(2, rnn),
            rtrans = PolyTransform(2, rnn),
            ztrans = IdTransform(),
         )

   # put together the basis specification

   # extract the maximum degrees


   # generate the scalar polynoials
   P0 = transformed_jacobi(deg0, r0trans, rcut0)
   Pr = transformed_jacobi(degr, rtrans, rcutr, rinr; pin = 2)
   Pθ = FourierBasis(degθ, Float64)
   Pz = transformed_jacobi(degz, ztrans, rcutz, -rcutz; pin = 2)

   # put together the basis
   return EnvPairBasis(P0, Pr, Pθ, Pz, alist, aalist)
end

function EnvPairBasis(N, deg, P0, Pr, Pz; kwargs...)

end
