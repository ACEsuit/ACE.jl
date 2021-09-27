
using ACE, Test, ACEbase, ACEbase.Testing

##

@info("Testing Onehot1pBasis")

@info "Test SList"

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

    # @info("check evaluation")
    B1p = ACE.Onehot1pBasis(categories; varsym = :mu, idxsym = :q)
    print_tf(@test ACE._varsym(B1p) == :mu)
    print_tf(@test ACE._isym(B1p) == :q)
    print_tf(@test ACE.symbols(B1p) == [:q,])
    print_tf(@test ACE.indexrange(B1p) == Dict(:q => categories))
    print_tf(@test length(B1p) == length(categories))


    EE = [true false false; false true false; false false true]
    
    for i = 1:len 
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

    ##

    # @info("check FIO")
    print_tf(@test all(ACEbase.Testing.test_fio(B1p)))
    println()
end