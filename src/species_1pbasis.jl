
abstract type SpeciesBasis end

struct Species1PBasisCtr{NZ} <: SpeciesBasis
   zlist::SZList{NZ}
end

struct Species1PBasisNeig{NZ} <: SpeciesBasis
   zlist::SZList{NZ}
end


zlist(basis::SpeciesBasis) = basis.zlist

Species1PBasisCtr(species) = Species1PBasisCtr(ZList(species, static=true))
Species1PBasisNeig(species) = Species1PBasisNeig(ZList(species, static=true))

Base.length(basis::Species1PBasis) = length(basis.zlist)^2

evaluate!(B, tmp, basis::Species1PBasisCtr, Xj, Xi) = evaluate!(B, tmp, basis, Xi)
evaluate!(B, tmp, basis::Species1PBasisCtr, Xj, Xi) = evaluate!(B, tmp, basis, Xj)

function evaluate!(B, tmp, basis::Species1PBasis{NZ}, X) where {NZ}
   fill!(B, 0)
   B[z2i(X.mu.z)] = 1
   return B
end

outtype(::SpeciesBasis) = Bool
