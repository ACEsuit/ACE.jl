
"""
Given a bond (i, j) it has an environment {k} which could be 
cylindrical, ellipsoidal, ...  
The ball B(ri, rcut_env) must be a superset of this environment.
"""
function cutoff_env end 


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
   r0cut::T   # bond-length cutoff
   rcut::T    # env cutoff into r-direction
   zcut::T    # 
   p0::Int
   pr::Int
   pz::Int
   floppy::Bool
   λ::T
end

CylindricalBondEnvelope(r0cut, rcut, zcut; p0 = 2, pr = 2, pz = 2, floppy = true, λ = .5) = 
      CylindricalBondEnvelope(r0cut, rcut, zcut, p0, pr, pz, floppy, λ)

cutoff_env(env::CylindricalBondEnvelope) = sqrt(env.rcut^2 + (env.r0cut + env.zcut)^2)
cutoff_radialbasis(env::CylindricalBondEnvelope) = sqrt(env.rcut^2 + (env.zcut + env.floppy * env.λ * env.r0cut)^2)

struct EllipsoidBondEnvelope{T} <: BondEnvelope{T}
   r0cut::T # bond-length cutoff
   rcut::T
   zcut::T  # must satisfy zcut >= r0cut/2
   p0::Int
   pr::Int
   floppy::Bool
   λ::T
end

EllipsoidBondEnvelope(r0cut, zcut, rcut; p0=2, pr=2, floppy=false, λ=.5) = EllipsoidBondEnvelope(r0cut, zcut, rcut, p0, pr, floppy, λ)
EllipsoidBondEnvelope(r0cut, cut; p0=2, pr=2, floppy=false, λ=.5) = EllipsoidBondEnvelope(r0cut, cut, cut, p0, pr, floppy, λ)

cutoff_env(env::EllipsoidBondEnvelope) = env.zcut + env.rcut 
cutoff_radialbasis(env::EllipsoidBondEnvelope) = env.zcut + env.floppy * env.λ * env.r0cut

function _evaluate_bond(env::BondEnvelope, X::AbstractState)
   r = norm(X.rr0)
   return ((r/env.r0cut)^2 - 1)^(env.p0) * (r <= env.r0cut)
end

function _evaluate_env(env::BondEnvelope, X::AbstractState)
   # convert to cylindrical coordinates
   z, r = cat2cyl(env, X)
   zeff = _eff_zcut(env, X)
   # then return the correct cutoff 
   return _eval_env_inner(env, z, r, zeff)
end

"""
`cat2cyl`: Convert cartesian coordinate to cylindrical coordinate
Given `env` that contains information of the centre and `X` the state
return Cylindrical coordinate of X as `z, r = cat2cyl(env, X)` 
"""
function cat2cyl(env::BondEnvelope, X::AbstractState)
   x_centre = env.λ * X.rr0
   x̃ = X.rr - x_centre
   if norm(X.rr0) >0
      r̂b = X.rr0/norm(X.rr0)
      z̃ = dot(x̃, r̂b)
      r̃ = norm( x̃ - z̃ * r̂b )
   else
      z̃ = 0.0
      r̃ = norm(x̃)
   end
   return z̃, r̃
end

# Compute effective zcut when `floppy = true` (to enable the cutoff in z direction to grow with the bond)
_eff_zcut(env::BondEnvelope, X::AbstractState) = env.zcut + env.floppy * norm(env.λ * X.rr0)

# The envelope for environment bonds
_eval_env_inner(env::CylindricalBondEnvelope, z, r, zeff) = ( (z/zeff)^2 - 1 )^(env.pz) * (abs(z) <= zeff) * 
         ( (r/env.rcut)^2 - 1 )^(env.pr) * (r <= env.rcut)

_eval_env_inner(env::EllipsoidBondEnvelope, z, r, zeff) = ( (z/zeff)^2 +(r/env.rcut)^2 - 1.0)^(env.pr) * ((z/zeff)^2 +(r/env.rcut)^2 <= 1)
