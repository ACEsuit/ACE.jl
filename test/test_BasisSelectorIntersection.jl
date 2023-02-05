using ACE, Test, ACEbase, ACEbase.Testing, StaticArrays
using ACE.Random: rand_rot, rand_refl
using Random: shuffle


r0cut = 2.0
rcut = 1.0
zcut = 2.0
env = ACE.EllipsoidBondEnvelope(r0cut, rcut, zcut;floppy=false, λ= .5)


@info( "Very basic test for intersection")
for maxorder = [1,2,3,4]

    Bsel = ACE.SparseBasis(; maxorder=maxorder, p = 2, default_maxdeg = 4) 


    RnYlm = ACE.Utils.RnYlm_1pbasis(;   r0 = ACE.cutoff_radialbasis(env), 
                                            rin = 0.0,
                                            trans = PolyTransform(1, ACE.cutoff_radialbasis(env)), 
                                            pcut = 0,
                                            pin = 0, Bsel = Bsel
                                        )

    basis = ACE.Utils.SymmetricBond_basis(ACE.Invariant(), env, Bsel;RnYlm=RnYlm )

    Bsel2 = ACE.CategorySparseBasis(:be, [:bond, :env]; maxorder = ACE.maxorder(Bsel), 
                                        p = Bsel.p, 
                                        weight = Bsel.weight, 
                                        maxlevels = Bsel.maxlevels,
                                        minorder_dict = Dict( :bond => 1),
                                        maxorder_dict = Dict( :bond => 1),
                                        weight_cat = Dict(:bond => 1.0, :env=> 1.0) 
                                    )


    BselIntersection = intersect(Bsel,Bsel2)

    Bc = ACE.Categorical1pBasis([:bond, :env]; varsym = :be, idxsym = :be )
    B1p =  Bc * RnYlm * env
    basis2 = ACE.SymmetricBasis(ACE.Invariant(), B1p, BselIntersection;)
    
    print_tf(@test length(basis) == length(basis2))

end