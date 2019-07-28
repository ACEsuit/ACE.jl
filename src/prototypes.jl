
# TODO [tuples]
# - get rid of `length_B`
# - document eval_basis and move it into JuLIP

import JuLIP: energy!, forces!, virial!, alloc_temp, alloc_temp_d
import JuLIP.Potentials: evaluate!, evaluate_d!


function eval_basis! end
function eval_basis_d! end

eval_basis(B, x, args...) =
   eval_basis!(alloc_B(B, x), alloc_temp(B, x), B, x, args...)

function eval_basis_d(B, x, args...)
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
