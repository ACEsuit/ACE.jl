using ACE
using ACEbase: precon!
using Flux: Optimise
using JuLIP
using ACE, ACEbase
using LinearAlgebra, Random
using ACE: evaluate, SymmetricBasis, NaiveTotalDegree, PIBasis
using Zygote
using Plots
using Optim
using StatsBase
using Flux
using Distributions

# ------------------------------------------------------------------------
#    site energies
# ------------------------------------------------------------------------
#to increase the data change NN
NN = 3

#train data
emt = EMT()
at = bulk(:Cu, cubic=true) * NN #14 gives 10976
rattle!(at,0.6)
nlist = neighbourlist(at, cutoff(emt))
train = []
for i = 1:length(at)
          Js, Rs, Zs = JuLIP.Potentials.neigsz(nlist, at, i); z0 = at.Z[i]
          Ei = JuLIP.evaluate(emt, Rs, Zs, z0)
          push!(train, (Rs=ACEConfig([EuclideanVectorState(Rs[j],"r") for j in 1:length(Rs)]), Ei=Ei))
end

#test data
test_size = length(at)/4
at = bulk(:Cu, cubic=true) * NN
rattle!(at,0.6)
nlist = neighbourlist(at, cutoff(emt))
test = []
for i = 1:Int(test_size)
          Js, Rs, Zs = JuLIP.Potentials.neigsz(nlist, at, i); z0 = at.Z[i]
          Ei = JuLIP.evaluate(emt, Rs, Zs, z0)
          push!(test, (Rs=ACEConfig([EuclideanVectorState(Rs[j],"r") for j in 1:length(Rs)]), Ei=Ei))
end

@show length(test)/(length(train)+length(test))

# ------------------------------------------------------------------------
#    basis
# ------------------------------------------------------------------------

# construct the 1p-basis
#to increase basis size simply increase this numbers. A good idea is to increase
#maxdeg to like 12 and keep ord to 5, but this will be VERY SLOW.
maxdeg = 5;
ord = 4;

D = NaiveTotalDegree();
B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, D = D);
φ = ACE.Invariant();
pibasis = PIBasis(B1p, ord, maxdeg; property = φ);
basis = SymmetricBasis(pibasis, φ);
BB = evaluate(basis, train[1].Rs)

# ------------------------------------------------------------------------
#    FinnisSinclair outer nonlinearity
# ------------------------------------------------------------------------
#to fiddle with this outter nonlinearity you need a new structure in ACE.Models.gfs
fun = ACE.Models.FinnisSinclair(1/10)
p = rand(length(BB),2) .- 0.5 #notice the 2 indicates the number of models, it's a matrix
FS = ACE.Models.GfsModel(basis, p, evaluator = :standard, F = fun) 

# ------------------------------------------------------------------------
#    Optimization functions
# ------------------------------------------------------------------------

#loss
λ = 0.0001
L_FS(θ,indx) = norm([ACE.Models.EVAL_me(FS,train[i].Rs)(θ) - train[i].Ei for i in indx],2)/length(indx) #+ λ*norm(θ,2)^2

#a function to save values from Optim.jl
function callBack(os)
    append!(tst_L,L_test(os.metadata["x"]))
    append!(trn_L, L_FS(os.metadata["x"],1:length(train)))
    append!(grd_n,os.g_norm)
    return false
end
L_test(θ) = norm([ACE.Models.EVAL_me(FS,test[i].Rs)(θ) - test[i].Ei for i in 1:length(test)],2)/length(test)


######################### Optim

function g!(G, x)
    gT = [getproperty.(Zygote.gradient(θ->L_FS(θ,1:length(train)),x)[1][i],:val) for i in 1:length(x[1,:])]
    G[:,1] = gT[1]
    G[:,2] = gT[2]
end


######################### FLUX


function opt_Flux(θ_S2, opt, b, iter)
    #saving information
    trn_loss = []
    tst_loss = []
    gradN = []
    #first append
    append!(trn_loss, L_FS(θ_S2,1:length(train)))
    append!(tst_loss, L_test(θ_S2))
    #our batch size, saved as a mutable array of length b
    indx = zeros(Int64,b)
    #n is here to double append the last grad, a small hack
    n=0
    for _ in 1:iter
        #we sample our training data and get the gradient
        StatsBase.sample!(1:length(train), indx; replace=false)
        g = Zygote.gradient(θ -> L_FS(θ,indx),θ_S2)[1]

        #flatten grad so it's easy to use
        gs = [collect(Iterators.flatten(g))[i].val for i in 1:length(θ_S2)]
        
        Flux.Optimise.update!(opt, θ_S2, gs)

        append!(trn_loss, L_FS(θ_S2,1:length(train)))
        n=norm(gs,1)
        append!(gradN,n)
        append!(tst_loss, L_test(θ_S2))

    end
    append!(gradN, n)
    return (θ_S2, ls_SGD2, gradN, tst_loss)
end

######################### precon

function opt_Flux_precon(θ_S2, opt, b, iter)
    ls_SGD2 = []
    tst_loss = []
    gradN = []
    append!(ls_SGD2, L_FS(θ_S2,1:length(train)))
    indx = zeros(Int64,b)

    P = Matrix(I, size(θ_S2)[1], size(θ_S2)[1])
    Q = cholesky(P).U

    for _ in 1:iter
        StatsBase.sample!(1:length(train), indx; replace=false)
        
        gs(θ_s) = [collect(Iterators.flatten(Zygote.gradient(θ -> L_FS(θ,indx),θ_s)[1]))[i].val for i in 1:length(θ_s)]

        du = rand(Normal(), size(θ_S2)[1])*sqrt(eps())
        g = gs(θ_S2); dg = gs(θ_S2 + du) - g

        rho = sqrt(maximum(abs.(du) * maximum(abs.(dg))))
        if(rho==0)
            @show rho
        end
        if (rho != 0)
            dg = dg/rho; du = du/rho
            t1 = Q*dg; t2 = du'/Q
            ∇E = UpperTriangular(t1*t1' - t2'*t2)
            α_Q = 0.1
            Q = Q - (α_Q/(maximum(abs.(∇E))+eps()))*∇E*Q
        end
        s = Q'*Q*g
        Flux.Optimise.update!(opt, θ_S2, s)

        append!(ls_SGD2, L_FS(θ_S2,1:length(train)))
        append!(tst_loss, L_test(θ_S2))
        n=norm(g,1)
        append!(gradN,n)

    end
    return (θ_S2, ls_SGD2, gradN, tst_loss)
end

# ------------------------------------------------------------------------
#    Optimization step
# ------------------------------------------------------------------------

@show "begin opt"

#initialize parameters, notice some of the optimizers work "in place" so we
#have to do a new one for each optimizer. We use a random seed so they are the same.
Random.seed!(1234)
θ_1 = rand(length(BB),2) .- 0.5
Random.seed!(1234)
θ_2 = rand(length(BB),2) .- 0.5
Random.seed!(1234)
θ_3 = rand(length(BB),2) .- 0.5

#adjust the learning rate h, the batch size b and the iteration number.
h = 0.1
b = 10 #use length(train) for full gradient
iter = 100

#we initialize the vectors to save our losses and gradient norm.
trn_L = []
tst_L = []
grd_n = []
lbfgs = @elapsed p1 = optimize(θ->L_FS(θ,1:length(train)), g!, θ_1, LBFGS(), Optim.Options(extended_trace=true, callback=callBack, store_trace = false, iterations = iter))
@show lbfgs
#simply plot these 3 agains iterations or time
#lbfgs is how long it took, and p1 still has some cool things, but nothing too important.
lbfgs_test = deepcopy(tst_L)
lbfgs_train = deepcopy(trn_L)
lbfgs_grad = deepcopy(grd_n)

#stochastic gradient descent
opt = Descent(h)
sgd = @elapsed p2 = opt_Flux(θ_2[:],opt,b,iter)
#p2[1] returns the params
#p2[2] return the training loss 
#p2[3] return the gradient norm (max norm as of now)
#p2[4] return the test loss
#sgd is the time the algorithm took
@show sgd

#ADAM with batching and preconditioning
opt = ADAM(h,(0.9, 0.999))
adam = @elapsed p3 = opt_Flux_precon(θ_3[:],opt,b,iter)
@show adam

# you can use all optimizers in Flux.jl and Optim.jl

#Flux algorithms

# SGD
# Momentum
# Nesterov
# RMSProp
# ADAM
# RADAM
# AdaMax
# OADAM
# ADAGrad
# ADADelta
# AMSGrad
# NADAM
# ADAMW
# ADaBelief

#Flux other fun things
# mix Optimise
# Decay
# clipping


####    PLOT    ITERATIONS    #### 
plot(p2[4], yaxis=:log, label="SGD", xlabel = "iterations", ylabel = "test Loss")
plot!(lbfgs_test,label="LBFGS")
plot!(p3[4],label="ADAM precon")

#savefig("testLoss.png")






