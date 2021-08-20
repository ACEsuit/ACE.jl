using LinearAlgebra: length
using ACE, ACEbase, Test, ACE.Testing
using ACE: evaluate, SymmetricBasis, NaiveTotalDegree, PIBasis, O3 
using StaticArrays


##
    
@info(" Testset for Multiple Properties in a Linear ACEModel")

# construct the 1p-basis
D = NaiveTotalDegree()
maxdeg = 6
ord = 3

B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, D = D)

# generate a configuration
nX = 54
Xs = () -> ACE.State(rr = rand(SVector{3, Float64}), u = rand())
cfg = ACEConfig([Xs() for i in 1:nX])

φ = ACE.Invariant()
pibasis = PIBasis(B1p, O3(), ord, maxdeg; property = φ)
basis = SymmetricBasis(φ, O3(), pibasis)

##

BB = evaluate(basis, cfg)

c_m = rand(SVector{7,Float64}, length(BB))

singlProp = [ACE.LinearACEModel(basis, rand(length(BB)), evaluator = :standard) for i in 1:length(c_m[1])]
multiProp = ACE.LinearACEModel(basis, c_m, evaluator = :standard)

@info("set_params!")
c_s = [[c_m[j][i] for j in 1:length(c_m)] for i in 1:length(c_m[1])]

ACE.set_params!(multiProp,c_m)
for i in 1:length(c_m[1])
    ACE.set_params!(singlProp[i],c_s[i])
    print_tf(@test(c_s[i] ≈ singlProp[i].c))
end
println()

##

@info("evaluate")

for i in 1:length(c_m[1])
    print_tf(@test(evaluate(singlProp[i],cfg).val ≈ evaluate(multiProp,cfg)[i].val))
end
println()


@info("grad_params")

multiGradP = ACE.grad_params(multiProp,cfg)

for i in 1:length(c_m[1])
    singl = getproperty.(ACE.grad_params(singlProp[i],cfg),:val)
    multi = [getproperty(multiGradP[j][i], :val) for j in 1:length(c_m)]
    print_tf(@test(singl ≈ multi))
end
println()

##

@info("grad_config")

for i in 1:length(c_m[1])
    singl = ACE.grad_config(singlProp[i],cfg)
    multi = ACE.grad_config(multiProp,cfg)[:,i]

    print_tf(@test(singl ≈ multi))
end
println()

##


# @info("grad_params_config")

# for i in 1:length(c_m[1])
#     singl = ACE.grad_params_config(singlProp[i],cfg)[1]
#     multi = ACE.grad_params_config(multiProp,cfg)[i]

#     print_tf(@test(singl ≈ multi))
# end
# println()
