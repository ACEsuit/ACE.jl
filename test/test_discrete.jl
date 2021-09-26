
using ACE, Test, ACEbase

##

@info("Testing Onehot1pBasis")

@info "Test SList"


categories = [:Si, :O, :Ti]
list = ACE.SList(categories)

for i = 1:3 
    println(@test ACE.val2i(list, categories[i]) == i)
    println(@test ACE.i2val(list, i) == categories[i])
end

##

@info("check evaluation")
Zmu = ACE.Onehot1pBasis(categories, :mu, :q)
println(@test ACE.symbols(Zmu) == [:q,])
println(@test ACE.indexrange(Zmu) == Dict(:q => categories))
println(@test length(Zmu) == length(categories))

for i = 1:3 
    ee(i) = [true false false; false true false; false false true][:, i]
    local X = ACE.State(mu = categories[i])
    println(@test ACE.evaluate(Zmu, X) == ee(i))
end

##

@info("check FIO")
println(@test all(ACEbase.Testing.test_fio(Zmu)))