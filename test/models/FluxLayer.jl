using Flux
using Statistics
using Flux: @epochs
using Flux: @functor
using ACE
using ACE: evaluate, SymmetricBasis, NaiveTotalDegree, PIBasis


# ------------------------------------------------------------------------
#    ACELayer
# ------------------------------------------------------------------------

#the activation funciton is inside GNL for now
struct ACELayer{T,TG}
    weights::Array{T,2}
    GNL::TG
end

function ACELayer(
    initW,
    σ,
    maxdeg,
    ord;
    totDeg = NaiveTotalDegree())

    B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, D = totDeg)
    φ = ACE.Invariant()
    pibasis = PIBasis(B1p, ord, maxdeg; property = φ)
    basis = SymmetricBasis(pibasis, φ);
    
    GNL = ACE.Models.NLModel(basis, initW, evaluator = :standard, F = σ) 
 
    ACELayer(initW, GNL)
end

@functor ACELayer

#rr are distances and X is some featurized matrix that will be fed into the next layer
function (l::ACELayer)(rr)
    #for now we ignore X and only use rr
    out_mat = [ACE.EVAL(l.GNL.LM[i],rr)(l.weights[:,i]) for i=1:length(l.weights[1,:])]
    rr, out_mat
end


# ------------------------------------------------------------------------
#    Pooling layer, calculate energies
# ------------------------------------------------------------------------

#the activation funciton is inside GNL for now
struct ACEPool{T,TG}
    weights::Array{T,2}
    GNL::TG
end

function ACEPool(
    initW,
    σ,
    maxdeg,
    ord;
    totDeg = NaiveTotalDegree())

    B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, D = totDeg)
    φ = ACE.Invariant()
    pibasis = PIBasis(B1p, ord, maxdeg; property = φ)
    basis = SymmetricBasis(pibasis, φ);
    
    GNL = ACE.Models.NLModel(basis, initW, evaluator = :standard, F = σ) 

    ACEPool(initW, GNL)
end

@functor ACEPool

#this layer joins an ACE layer plus a pooling layer
function (l::ACEPool)(data)
    (rr, X) = data
    #for now we ignore X and only use rr
    out = ACE.Models.EVAL_NL(l.GNL,rr)(l.weights)
    out
end


# ------------------------------------------------------------------------
#    Sample model of 2 FS layers
# ------------------------------------------------------------------------

maxdeg = 5;
ord = 4;
nX = 54

D = NaiveTotalDegree();
B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, D = D);
φ = ACE.Invariant();
pibasis = PIBasis(B1p, ord, maxdeg; property = φ);
basis = SymmetricBasis(pibasis, φ);

#training data (site energies for now)
train_cfgs = [ACEConfig(rand(EuclideanVectorState, B1p.bases[1], nX)) for i in 1:80]
train_Ei =  [sum(evaluate(basis,train_cfgs[i])).val for i in 1:80]
test_cfgs =  [ACEConfig(rand(EuclideanVectorState, B1p.bases[1], nX)) for i in 1:15]
test_Ei =  [sum(evaluate(basis,test_cfgs[i])).val for i in 1:15]

train_data = zip(train_cfgs, train_Ei)

σ = ϕ -> ϕ[1] + sqrt((1/10)^2 + abs(ϕ[2])) - 1/10
BB = evaluate(basis, ACEConfig(rand(EuclideanVectorState, B1p.bases[1], nX)))
initW = rand(length(BB),2).-0.5 

model = Chain(ACELayer(initW,σ,maxdeg,ord),ACEPool(initW,σ,maxdeg,ord))

loss(x, y) = Flux.Losses.mse(model(x), y)

opt = ADAM(0.3)
num_epochs = 10 # how many epochs to train?

# and a callback to see training progress
evalcb() = @show(mean(loss.(test_cfgs, test_Ei)))
evalcb()

# train
println("Training!")
@epochs num_epochs Flux.train!(
    loss,
    params(model),
    train_data,
    opt,
    cb = Flux.throttle(evalcb, 10),
)