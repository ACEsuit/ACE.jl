

export ProdPIBasis

mutable struct ProdPIBasis{NB, REAL} <: ACEBasis
   bases::NTuple{NB, PIBasis}
   spec::Vector{NTuple{NB, Int}}
   real::REAL     # could be `real` or `identity` to keep AA complex
   # evaluator    # classic vs graph
end

import Base.*

*(B1::PIBasis, B2::PIBasis) =
   ProdPIBasis( (setreal(B1, false), setreal(B2, false)),
                Matrix{Int}(undef, (0,2)), Base.real )

function ProdPIBasis(bases, real=Base.real;
                     Deg = nothing, maxdeg = nothing)

end
