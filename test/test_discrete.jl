
using ACE, Test, ACEbase, ACEbase.Testing, 
      StaticArrays

using ACE.Random: rand_rot, rand_refl
using Random: shuffle 
using LinearAlgebra: norm 
##

@info("Testing Categorical1pBasis")

@info("Running some basic evaluation checks")

for categories in (  [:a,], 
                     [:a, :b, :c], 
                     [1, 2], 
                     [true, false] )
    @info("categories = $categories")
    len = length(categories) 

    list = ACE.SList(categories)

    for i = 1:len  
        print_tf(@test ACE.val2i(list, categories[i]) == i)
        print_tf(@test ACE.i2val(list, i) == categories[i])
    end

    ##
    local B1p
    # @info("check evaluation")
    B1p = ACE.Categorical1pBasis(categories; varsym = :mu, idxsym = :q)
    print_tf(@test ACE._varsym(B1p) == :mu)
    print_tf(@test ACE._isym(B1p) == :q)
    print_tf(@test ACE.symbols(B1p) == [:q,])
    print_tf(@test ACE.indexrange(B1p) == Dict(:q => categories))
    print_tf(@test length(B1p) == length(categories))

    local EE 
    EE = [true false false; false true false; false false true]

    for i = 1:len 
        local X 
        ee(i) = EE[1:len, i]
        X = ACE.State(mu = categories[i])
        print_tf(@test ACE.evaluate(B1p, X) == ee(i))
    end

    # this throws an error 
    # X = ACE.State(mu = :x)
    # print_tf(@test all(ACE.evaluate(B1p, X) .== false) )

    # @info("check reading from basis ")

    for i = 1:len 
        b = (q = categories[i], )
        print_tf(@test ACE.degree(b, B1p) == 0) 
        print_tf(@test ACE._idx(b, B1p) == categories[i])
    end

    # @info("check FIO")
    print_tf(@test all(ACEbase.Testing.test_fio(B1p)))
    println()
end


##

Bsel = SimpleSparseBasis(3, 6)
B1p_be = ACE.Categorical1pBasis([:e, :b]; varsym = :be, idxsym=:be)
RnYlm = ACE.Utils.RnYlm_1pbasis(; Bsel=Bsel)
B1p = B1p_be * RnYlm
basis = ACE.SymmetricBasis(ACE.Invariant(), B1p, Bsel)
@show length(basis)

##

cfg = [ ACE.State(rr = rand_vec3(B1p["Rn"]), 
                  be = rand([:b,:e]) ) 
        for _ = 1:10 ] |> ACEConfig
B1 = ACE.evaluate(basis, cfg)

for ntest = 1:30
    Q = rand_refl() * rand_rot()
    Xs2 = shuffle([ ACE.State(rr = Q * X.rr, be = X.be) for X in cfg.Xs ])
    B2 = ACE.evaluate(basis, ACEConfig(Xs2))
    print_tf(@test isapprox(B1, B2, rtol=1e-10))
end
println()

# ##
# THIS TEST REALLY HAS NO PLACE HERE -- 
# DELETE IT SOMETIME IN THE FUTURE IF WE DONT REVIVE IT...
#

# @info("Test spherical covariance")

# L1 = L2 = 1 
# basis = ACE.SymmetricBasis(ACE.SphericalMatrix(L1,L2; T = ComplexF64), B1p, Bsel; 
#                            filterfun=ACE.NoConstant())
# @show length(basis)

# B1 = ACE.evaluate(basis, cfg)

# for ntest = 1:30
#     Q, D1, D2 = ACE.Wigner.rand_QD(L1, L2)
#     Xs2 = shuffle([ ACE.State(rr = Q * X.rr, be = X.be) for X in cfg.Xs ])
#     B2 = ACE.evaluate(basis, ACEConfig(Xs2))
#     D1txB1xD2 = Ref(D1') .* B2 .* Ref(D2)
#     print_tf(@test isapprox(D1txB1xD2, B1, rtol=1e-10))
# end
# println()

