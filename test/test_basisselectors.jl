using ACE


maxorder = 3
weight = Dict(:l =>1, :n => 1)
degree = Dict("default" => 5)
p = 1
BSel = SparseBasis(maxorder,weight,degree,p)

maxorder_dict = Dict(:bond => 1)
isym = :be
weight_cat = Dict(:env => 1, :bond => 1)

BSel_cat = ACE.CategoryBasisSelector(BSel.maxorder, BSel.weight, BSel.degree, BSel.p, isym, weight_cat, maxorder_dict)

Bc = ACE.Categorical1pBasis([:bond,:env]; varsym = isym, idxsym = isym)
RnYlm = ACE.Utils.RnYlm_1pbasis(; )

r0cut = 2.0
rcut = 1.0
zcut = 2.0
env = ACE.EllipsoidBondEnvelope(r0cut, rcut, zcut;floppy=false, λ= .5)

ϕ = ACE.Invariant()

B1p =  Bc * RnYlm
basis_inv_cat = ACE.SymmetricBasis(ϕ, B1p, BSel_cat)
@show length(basis_inv_cat)

Bsel_cat_weight = ACE.CategoryWeightedBasisSelector(BSel.weight, BSel.degree, BSel.p, isym, weight_cat )
Bsel_cat_int = ACE.CategoryConstraint(:be, maxorder_dict)
Bsel_ord_constr = ACE.OrderConstraint(maxorder)


BSel_cat2 =  ACE.intersect(Bsel_cat_weight,ACE.intersect(Bsel_temp,Bsel_ord_constr) )
basis_inv_cat2 = ACE.SymmetricBasis(ϕ, B1p, BSel_cat2)
@show length(basis_inv_cat2)

for b in BSel_cat2
    print(typeof(b),"\n")
end
#using ACE: ∩
#BSel_cat2 = Bsel_cat_weight :∩ Bsel_cat_int :∩ Bsel_ord_constr

ACE.maxorder(BSel_cat2) 