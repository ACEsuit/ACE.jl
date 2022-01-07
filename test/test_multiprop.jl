using LinearAlgebra: length
using ACE, ACEbase, Test, ACE.Testing
using ACE: evaluate, SymmetricBasis, PIBasis, O3 
using StaticArrays


##
    
@info(" Testset for Multiple Properties in a Linear ACEModel")

# construct the 1p-basis
maxdeg = 6
ord = 3
Bsel = SimpleSparseBasis(ord, maxdeg)

B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg)

# generate a configuration
#TODO check if this tests account/test u, and will it work with more things?
nX = 54
Xs = () -> ACE.State(rr = rand(SVector{3, Float64}), u = rand())
cfg = ACEConfig([Xs() for i in 1:nX])

φ = ACE.Invariant()
basis = SymmetricBasis(φ, B1p, O3(), Bsel)

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

# We compare a model with a sinlge property, to the corresponding solution
# of the multiple property model. We do this by giving the same parameters
# to both models and then using isapprox 

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

@info("adjoint_EVAL_D 1 prop")

#we check by contracting the full matrix that adjoint_eval_config works
#this assumes that grad_params_config works for one porperty. This is 
#tested elsewhere. 
for i in 1:length(c_m[1])

    #find the jacobian
    Jac = ACE.grad_params_config(singlProp[i],cfg)

    #create a random input emulating the pullback input
    w = rand(SVector{3, Float64}, length(Jac[1,:]))
    u = zeros(1)
    w = [ACE.DState(rr = w[j], u = u) for j in 1:length(w)]

    #calculate the adjoint and make a zeros to fill
    grad = ACE.adjoint_EVAL_D(singlProp[i], cfg, w)
    Jgrad = zeros(length(grad))

    #contract w into the jacobian to get the solution
    for j in 1:length(Jac[:,1])
        Jgrad[j] = sum([ACE.contract(Jac[j,:][k], w[k]) for k in 1:length(w)])
    end

    print_tf(@test(grad ≈ Jgrad))
end
println()

@info("adjoint_EVAL_D >2 prop")

#now we check that multiple properties work. For this, like before, we simply
#compare the single property result to each of the multiple properties. 

uo = zeros(1)
function wMaker()
    wtmp = rand(SVector{3, Float64}, length(cfg))
    wtmp = [ACE.DState(rr = wtmp[j], u = uo) for j in 1:length(wtmp)]
    return wtmp
end
TDX1 = ACE.DState{NamedTuple{(:rr, :u), Tuple{SVector{3, Float64}, Vector{Float64}}}}
wo = Matrix{TDX1}(undef, (54,7))
wt  = [wMaker() for i in 1:length(c_m[1])]

for i in 1:length(wt)
    for j in 1:length(wt[i])
        wo[j,i] = wt[i][j]
    end
end

multiEval = ACE.adjoint_EVAL_D(multiProp, cfg, wo)

for i in 1:length(c_m[1])
    singl = ACE.adjoint_EVAL_D(singlProp[i], cfg, wo[:,i])
    multi = [multiEval[j][i] for j in 1:length(c_m)]
    print_tf(@test(singl ≈ multi))
end
println() 
