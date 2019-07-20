
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


# TODO [tuples]
# - get rid of `length_B` 

import JuLIP: energy!, forces!, virial!, alloc_temp, alloc_temp_d
import JuLIP.Potentials: evaluate!, evaluate_d!


function eval_basis! end
function eval_basis_d! end

eval_basis(B, x) =
   eval_basis!(alloc_B(B, x), alloc_temp(B), B, x)

function eval_basis_d(B, x)
   b = alloc_B(B, x)
   db = alloc_dB(B, x)
   tmp = alloc_temp_d(B, x)
   eval_basis_d!(b, db, tmp, B, x)
   return b, db
end

function alloc_B end
function alloc_dB end

function transform end
function transform_d end
function fcut end
function fcut_d end


# auxiliary stuff
@generated function nfcalls(::Val{N}, f) where {N}
   code = Expr[]
   for n = 1:N
      push!(code, :(f(Val($n))))
   end
   quote
      $(Expr(:block, code...))
      return nothing
   end
end
