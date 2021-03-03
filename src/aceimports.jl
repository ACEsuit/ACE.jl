
# --------------------------------------------------------------------------
# ACE.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------



import ACE

import ACE: ACEBasis, ScalarACEBasis, OneParticleBasis,
              PIBasis,
              get_spec,
              AbstractDegree, degree,
              gensparse,
              add_into_A!, add_into_A_dA!,
              scaling

# old stuff to delete or re-introduce ...
# , PIPotential, PIBasisFcn, site_alloc_B, site_alloc_dB,
# site_evaluate!, site_evaluate_d!,
# set_Aindices!, VecOrTup
