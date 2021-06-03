
using ACEbase: @def

@def aceimports begin

   import ACE

   import ACE:   PIBasis,
                 get_spec,
                 AbstractDegree, degree,
                 gensparse,
                 add_into_A!, add_into_A_dA!,
                 scaling, 
                 AbstractACEModel, 
                 AbstractProperty
end

@def baseimports begin
   import ACEbase

   import ACEbase: ACEBasis, ScalarACEBasis, OneParticleBasis, Discrete1pBasis, 
                   alloc_temp, alloc_temp_d, alloc_temp_dd,
                   fltype, rfltype, gradtype,
                   evaluate, evaluate_d, evaluate_dd, evaluate_ed,
                   evaluate!, evaluate_d!, evaluate_dd!, evaluate_ed!,
                   precon!, _allfieldsequal,
                   alloc_B, alloc_dB,
                   AbstractState,
                   AbstractConfiguration,
                   AbstractContinuousState,
                   AbstractDiscreteState


   import ACEbase: FIO
   import ACEbase.FIO: read_dict, write_dict, save_json, load_json
end


# modules external to our own eco-system, rigorously separate using and import

@def extimports begin

   using Parameters: @with_kw

   using Random: shuffle

   import Base: ==, length, kron, filter

   # for some reason it seems we need to export this to be used outside?!?!

   using LinearAlgebra: norm, dot, mul!, I

   using StaticArrays
end




# old stuff to delete or re-introduce ...
# , PIPotential, PIBasisFcn, site_alloc_B, site_alloc_dB,
# site_evaluate!, site_evaluate_d!,
# set_Aindices!, VecOrTup
