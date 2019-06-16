

function eval_basis end
function eval_basis! end
function eval_grad end
function eval_basis_d! end

eval_basis(B, args...) = eval_basis!(zeros(length(B)), args...)
eval_basis_d(B, args...) = eval_basis!(zeros(length(B)), zeros(length(B)), args...)

function transform end
function transform_d end
function fcut end
function fcut_d end
