


abstract type BondEnvelope{T} <: B1pMultiplier{T}  end 


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

function evaluate(env::BondEnvelope, X::AbstractState)
   if X.be == :env 
      return _evaluate_bond(env, X)
   elseif X.be == :bond 
      return _evaluate_env(env, X)
   else
      error("invalid X.be value")
   end
end

function filter(env::CylindricalBondEnvelope, X)
   # return true or false 
end

function _evaluate_bond(env::CylindricalBondEnvelope)
   r = norm(X.rr0)
   return (r - env.r0cut)^(env.p0)
end


function _evaluate_env(env::CylindricalBondEnvelope)
   # convert to cylindrical coordinates

   # then return the correct cutoff 

end

