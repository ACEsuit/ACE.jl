using ACE
using Zygote
using LinearAlgebra
import ChainRulesCore, ChainRules
import ChainRulesCore: rrule, NO_FIELDS 

@doc raw"""
A generalized non linear model.
```math
F(ϕ_1,ϕ_2,...,ϕ_N): R^N -> R^1
```
It contains (basis,Array{LinearACEModel},evaluator,F).
Basically it buils N linear models and runs them through
the nonlinearity F. 
params and set_params! interact directly with the respective
linear model.
F can be anything and is passed as a function taking a vector and
returning a single value.
Make sure the parameters you pass are in array of arrays form. So that
c[n] corresponds to the parameters of linear model n or ϕ_n.
"""

struct NLModel{TB, TLM, TEV, TF} 
   basis::TB
   LM::AbstractArray{TLM}
   evaluator::TEV
   F::TF
end

function set_params!(m::NLModel,c)
    for (i,lm) in enumerate(m.LM)
        ACE.set_params!(lm, c[i])
    end
end

params(m) = [params(lm) for lm in m.LM]


function NLModel(basis, c;
     evaluator=:standard, F=F) 
   if evaluator == :naive 
      ev = θ -> ACE.NaiveEvaluator()
   elseif evaluator == :standard 
      ev = θ -> ACE.PIEvaluator(basis, θ) 
   elseif evaluator == :recursive 
      error("Recursive evaluator not yet implemented")
   else 
      error("unknown evaluator")
   end
   LM = [ACE.LinearACEModel(basis, c[i], ev(c[i])) for i=1:length(c)]
   return NLModel(basis, LM, ev, F)
end

#simple wrappers around site energies
struct EVAL_NL{TM, TX}
    m::TM 
    X::TX
 end

function (y::EVAL_NL)(θ)
    return y.m.F([ACE.EVAL(y.m.LM[i],y.X)(θ[i]) for i=1:length(θ)])
end

#wrapper around forces. EVAL_D_NL returns the force of a site.
struct EVAL_D_NL{TM, TX}
    m::TM 
    X::TX
end

function (y::EVAL_D_NL)(θ)
    g(x) = Zygote.gradient(y.m.F, [ACE.EVAL(y.m.LM[i],y.X)(x[i]) for i=1:length(x)])[1]
    return sum([ACE.EVAL_D(y.m.LM[i],y.X)(θ[i])*g(θ)[i] for i=1:length(θ)])
end


#This is for multiplying special types
#given parameters and configs it returns an
#extended tensor of size (θ,X,3) 3 for ,xyz coordinates
function genMul(θ, X)
    T = typeof(X[1])
    prod = Array{T}(undef, (size(θ)[1], size(X)[1]))
    for (i,t) in enumerate(θ)
       for (j,x) in enumerate(X)
          prod[i,j] = t .* x
       end
    end
    return(prod)
 end

 #this works like the contraction of the product basis.
 #it could be incorporated into genMul for faster evaluation.
 #it could maybe even use a smarter evaluation (recursive?).
function muldXdθ(dB, w)    
    g = zeros(size(dB, 1))
    for i = 1:length(g), j = 1:size(dB, 2)
       g[i] += dot(dB[i, j], w[j])
    end
    return g
end


 #maybe throw in an if to skip computations when h[i,j]=0
 #Place everything inside the adjoint? how many adjoint calls? should be 1 only.
 
 #The rrule for the derivative of forces according to the gradient.
 #We use zygote to get the hessian and the gradient of our nonlinearity. 
 #then we need 3 things, dθX, dθ and dX. We simply construct the adjoint
 #"manually" by multiplying these elements accordingly.
 function rrule(y::EVAL_D_NL, θ)
    at = [ACE.EVAL(y.m.LM[i],y.X)(θ[i]) for i=1:length(θ)]
    #derivative of nonlinearity and hessian
    g_nl = Zygote.gradient(y.m.F, at)[1]
    h = Zygote.hessian(y.m.F, at)
    #derivative of EVAL_D calls the smart adjoint
    dXθ = [Zygote.pullback(ACE.EVAL_D(y.m.LM[i],y.X),θ[i])[2] for i=1:length(θ)] 
    dX = [ACE.grad_config(y.m.LM[i], y.X) for i=1:length(θ)] 
    dθ = [ACE.getproperty.(ACE.grad_params(y.m.LM[i], y.X), :val) for i=1:length(θ)] 
    
    #product rule
    function adj(dp)

       sol = [dXθ[i](dp)[1] .* g_nl[i] + 
                sum([h[j,i] .* muldXdθ(genMul(dθ[i],dX[j]),dp) for j=1:size(h,1)])
                 for i=1:length(θ)]

       return ( NO_FIELDS ,  sol )       
    end

    val = sum([ACE.EVAL_D(y.m.LM[i],y.X)(θ[i])*g_nl[i] for i=1:length(θ)])

    return val, adj
 end

# ------------------------------------------------------------------------
#    Codes for Energies and Forces
# ------------------------------------------------------------------------

#This should not be in ACE anyways, but for now we create an energy object that 
#includes the model and the series of configurations in an atom object. It acts as
#a calculator such that energy(θ) will return the energy with θ parameters.
using JuLIP

struct ENERGY_NL{TM, TRS}
    m::TM 
    Rs::TRS
end

function ENERGY_NL(m, at, vref)
    Rs = []
    nlist = neighbourlist(at, cutoff(vref))
    for i = 1:length(at)
        Js, tmpRs, Zs = JuLIP.Potentials.neigsz(nlist, at, i); z0 = at.Z[i]
        tmpRs=ACEConfig([EuclideanVectorState(tmpRs[j],"r") for j in 1:length(tmpRs)])
        append!(Rs,[tmpRs])
    end 
    return ENERGY_NL(m,Rs)
end

function (y::ENERGY_NL)(θ)
    return sum([EVAL_NL(y.m,r)(θ) for r in y.Rs])
end

struct FORCES_NL{TM, TRS}
    m::TM 
    Rs::TRS
end

#this structure if repeated with energy, maybe smart dispatching could save
#memory and time.
function FORCES_NL(m, at, vref)
    Rs = []
    nlist = neighbourlist(at, cutoff(vref))
    for i = 1:length(at)
        Js, tmpRs, Zs = JuLIP.Potentials.neigsz(nlist, at, i); z0 = at.Z[i]
        tmpRs=ACEConfig([EuclideanVectorState(tmpRs[j],"r") for j in 1:length(tmpRs)])
        append!(Rs,[tmpRs])
    end 
    return(FORCES_NL(m,Rs))
end

function (y::FORCES_NL)(θ)
    return [sum(EVAL_D_NL(y.m,r)(θ)) for r in y.Rs]
end
