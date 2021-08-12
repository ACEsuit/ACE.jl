using LinearAlgebra: length
using ACE, ACEbase, Test, ACE.Testing
using ACE: evaluate, SymmetricBasis, NaiveTotalDegree, PIBasis

using StaticArrays


##
    
# construct the 1p-basis
D = NaiveTotalDegree()
maxdeg = 6
ord = 3

B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, D = D)

# generate a configuration
nX = 54
Xs() = ACE.State(rr = rand(SVector{3, Float64}), u = rand())
cfg = ACEConfig([Xs() for i in 1:nX])
a(x) = fieldnames(typeof(x))

φ = ACE.Invariant()
pibasis = PIBasis(B1p, ord, maxdeg; property = φ)
basis = SymmetricBasis(pibasis, φ)
       
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

@info("evaluate")

for i in 1:length(c_m[1])
    print_tf(@test(evaluate(singlProp[i],cfg).val ≈ evaluate(multiProp,cfg)[i].val))
end
println()


@info("grad_params")
#for now grad_params only gets B ONCE, we could copy it multiple times.
for i in 1:length(c_m[1])
    print_tf(@test(getproperty.(ACE.grad_params(singlProp[i],cfg),:val) ≈ getproperty.(ACE.grad_params(multiProp,cfg)[:,i],:val)))
end
println()


@info("grad_config")

function config_dist_eq(A,B)
    pass = true
    for i in 1:length(A)
        if(A[i].rr != B[i].rr)
            pass = false
        end
    end
    return pass
end

for i in 1:length(c_m[1])
    singl = ACE.grad_config(singlProp[i],cfg)
    multi = ACE.grad_config(multiProp,cfg)[:,i]

    print_tf(@test(config_dist_eq(singl, multi)))
end
println()

@info("grad_params_config")

for i in 1:length(c_m[1])
    singl = ACE.grad_params_config(singlProp[i],cfg)[1]
    multi = ACE.grad_params_config(multiProp,cfg)[i]

    print_tf(@test(config_dist_eq(singl, multi)))
end
println()
