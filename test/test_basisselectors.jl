using ACE, Test, ACEbase, ACEbase.Testing, StaticArrays
using ACE.Random: rand_rot, rand_refl
using Random: shuffle


#=
@info("Rudimentary tests for sparse basis selectors and intersections of such")
r0cut = 2.0
rcut = 1.0
zcut = 2.0
env = ACE.EllipsoidBondEnvelope(r0cut, rcut, zcut;floppy=false, λ= .5)

maxorder = 3
dmaxdeg = 4
Bsel_p2 = ACE.PNormSparseBasis(maxorder; p = 2, default_maxdeg = dmaxdeg) 
Bsel_p1 = ACE.PNormSparseBasis(maxorder; p = 1, default_maxdeg = dmaxdeg) 

Bsel_intesect = ACE.intersect(Bsel_p1,Bsel_p2)
B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=dmaxdeg)
basis_intersect_inv = ACE.SymmetricBasis(ACE.Invariant(), B1p, Bsel_intesect)
=#

@info("Test bond basis selectors")

r0cut = 2.0
rcut = 1.0
zcut = 2.0
env = ACE.EllipsoidBondEnvelope(r0cut, rcut, zcut;floppy=false, λ= .5)

maxorder = 3
Bsel = ACE.PNormSparseBasis(maxorder; p = 2, default_maxdeg = 4) 


@info("Test invariance")


basis_inv = ACE.Utils.SymmetricBond_basis(ACE.Invariant(), env, Bsel; )
@show length(basis_inv)

rr0 = SVector{3}(rand(Float64,3))
cfg = [ ACE.State(rr = SVector{3}(rand(Float64,3)), rr0 = rr0,
                  be = rand([:bond,:env])) 
        for _ = 1:10 ] |> ACEConfig
B1_inv = ACE.evaluate(basis_inv, cfg)

for ntest = 1:30
    Q = rand_refl() * rand_rot()
    Xs2 = ACE.shuffle([ ACE.State(rr = Q * X.rr, rr0 = Q * X.rr0, be = X.be)  for X in cfg.Xs ])
    B2_inv = ACE.evaluate(basis_inv, ACEConfig(Xs2))
    print_tf(@test isapprox(B1_inv, B2_inv, rtol=1e-10))
end
println()
@info("Test Euclidian covariance")

basis_cov = ACE.Utils.SymmetricBond_basis(ACE.EuclideanVector(), env, Bsel; )
@show length(basis_cov)

rr0 = SVector{3}(rand(Float64,3))
cfg = [ ACE.State(rr = SVector{3}(rand(Float64,3)), rr0 = rr0,
                  be = rand([:bond,:env])) 
        for _ = 1:10 ] |> ACEConfig
B1_cov = ACE.evaluate(basis_cov, cfg)


for ntest = 1:30
    Q = rand_refl() * rand_rot()
    Xs2 = ACE.shuffle([ ACE.State(rr = Q * X.rr, rr0 = Q * X.rr0, be = X.be)  for X in cfg.Xs ])
    B2_cov = ACE.evaluate(basis_cov, ACEConfig(Xs2))
    print_tf(@test isapprox( map(x->Q*x, B1_cov), B2_cov, rtol=1e-10))
end
println()