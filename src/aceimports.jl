
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------



import ACE

import ACE: ScalarBasis, OneParticleBasis, OnepBasisFcn,
              PIBasis, PIPotential, PIBasisFcn,
              VecOrTup, get_basis_spec,
              AbstractDegree, degree,
              gensparse,
              site_alloc_B, site_alloc_dB,
              site_evaluate!, site_evaluate_d!,
              set_Aindices!, add_into_A!, add_into_A_dA!,
              scaling
