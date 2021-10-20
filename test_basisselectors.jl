using ACE


maxorder = 3
weight = Dict(:l =>1, :n => 1)
degree = Dict("default" => 5)
p = 1
BSel = SparseBasis(maxorder,weight,degree,p)

maxorder_dict = Dict(:bond => 1)
isym = :be
weight_cat = Dict(:env => 1, :bond => 1)

BSel_cat = ACE.CategoryBasisSelector(BSel.maxorder, maxorder_dict, isym, BSel.weight, weight_cat, BSel.degree, BSel.p)
Bc = ACE.Categorical1pBasis([:bond,:env]; varsym = isym, idxsym = isym)
RnYlm = ACE.Utils.RnYlm_1pbasis(; )

r0cut = 2.0
rcut = 1.0
zcut = 2.0
env = ACE.EllipsoidBondEnvelope(r0cut, rcut, zcut;floppy=false, λ= .5)

ϕ = ACE.Invariant()

B1p =  Bc * RnYlm
basis_inv_cat = ACE.SymmetricBasis(ϕ, B1p, BSel_cat)

Bsel_cat_weight = CategoryWeightedBasisSelector(BSel.maxorder,  isym, BSel.weight, weight_cat, BSel.degree, BSel.p)
Bsel_cat_int = Category
basis_inv_cat_int = ACE.SymmetricBasis(ϕ, B1p, BSel ∩ Bsel_cat_weight ∩ Bsel_cat_int)
length(basis_inv)

