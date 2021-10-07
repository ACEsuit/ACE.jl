


abstract type BondEnvelope{T} <: B1pMultiplier{T}  end

# we can't call this `evaluate` since that would return a Vector
# whereas _inner_evaluate(...) should just give us a value.
function _inner_evaluate(env::BondEnvelope, X::AbstractState)
   if X.be == :env
      return _evaluate_env(env, X)
   elseif X.be == :bond
      return _evaluate_bond(env, X)
   else
      error("invalid X.be value")
   end
end

Base.filter(env::BondEnvelope, X::AbstractState) =
         _inner_evaluate(env,X) != 0

struct CylindricalBondEnvelope{T} <: BondEnvelope{T}
   r0cut::T
   rcut::T
   zcut::T
   p0::Int
   pr::Int
   pz::Int
end

CylindricalBondEnvelope(r0cut, rcut, zcut; p0 = 2, pr = 2, pz = 2) =
      CylindricalBondEnvelope(r0cut, rcut, zcut, p0, pr, pz)



function _evaluate_bond(env::CylindricalBondEnvelope, X::AbstractState)
   r = norm(X.rr0)
   return ((r/env.r0cut)^2 - 1)^(env.p0) * (r <= env.r0cut)
end


function _evaluate_env(env::CylindricalBondEnvelope, X::AbstractState)
   # convert to cylindrical coordinates
   r_centre = X.rr0/2
   r̂b = X.rr0/norm(X.rr0)
   z = dot(X.rr - r_centre, r̂b)
   r = norm( (X.rr - r_centre) - z * r̂b )
   # then return the correct cutoff
   zcuteff = env.zcut + norm(X.rr0) / 2
   return ( (z/zcuteff)^2 - 1 )^(env.pz) * (abs(z) <= zcuteff) *
            ( (r/env.rcut)^2 - 1 )^(env.pr) * (r <= env.rcut)
end


struct ElipsoidBondEnvelope{T} <: BondEnvelope{T}
   r0cut::T
   rcut::T
   zcut::T  # must satisfy zcut >= r0cut/2
   p0::Int
   pr::Int
   floppy::Bool
   λ::T
end

ElipsoidBondEnvelope(r0cut, zcut, rcut; p0=2, pr=2, floppy=false, λ=.5) = ElipsoidBondEnvelope(r0cut, zcut, rcut, p0, pr, floppy, λ)
ElipsoidBondEnvelope(r0cut, cut; p0=2, pr=2, floppy=false, λ=.5) = ElipsoidBondEnvelope(r0cut, cut, cut, p0, pr, floppy, λ)

function _evaluate_bond(env::ElipsoidBondEnvelope, X::AbstractState)
   r = norm(X.rr0)
   return ((r/env.r0cut)^2 - 1)^(env.p0) * (r <= env.r0cut)
end

function _evaluate_env(env::ElipsoidBondEnvelope, X::AbstractState)
   # convert to cylindrical coordinates
   x_centre = env.λ * X.rr0
   x̃ = X.rr - x_centre
   if norm(X.rr0) > 0
      x_centre_0  = X.rr0/norm(X.rr0)
      z̃ = dot(x_centre_0, x̃ )
      r̃ = norm( x̃ - z̃ * x_centre_0)
   else
      z̃ = 0.0
      r̃ = norm(x̃)
   end
   # then return the correct cutoff
   zeff = env.zcut + env.floppy * norm(x_centre)
   g = (z̃/zeff)^2 +(r̃/env.rcut)^2
   return ( g - 1.0)^(env.pr) * (g <= 1)
end
