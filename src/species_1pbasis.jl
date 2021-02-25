
abstract type SpeciesBasis{NZ} end

struct Species1PBasisCtr{NZ} <: SpeciesBasis{NZ}
   zlist::SZList{NZ}
end

struct Species1PBasisNeig{NZ} <: SpeciesBasis{NZ}
   zlist::SZList{NZ}
end


zlist(basis::SpeciesBasis) = basis.zlist

Species1PBasisCtr(species) = Species1PBasisCtr(ZList(species, static=true))
Species1PBasisNeig(species) = Species1PBasisNeig(ZList(species, static=true))

Base.length(basis::SpeciesBasis) = length(basis.zlist)^2

evaluate!(B, tmp, basis::Species1PBasisCtr, Xj, Xi) = evaluate!(B, tmp, basis, Xi)
evaluate!(B, tmp, basis::Species1PBasisNeig, Xj, Xi) = evaluate!(B, tmp, basis, Xj)

function evaluate!(B, tmp, basis::SpeciesBasis{NZ}, X) where {NZ}
   fill!(B, 0)
   B[z2i(basis.zlist, X.mu)] = 1
   return B
end

fltype(::SpeciesBasis) = Bool

symbols(::Species1PBasisCtr) = [:μ0]
symbols(::Species1PBasisNeig) = [:μ]

indexrange(basis::Species1PBasisCtr) = Dict( :μ0 => Int.(basis.zlist.list) )
indexrange(basis::Species1PBasisNeig) = Dict( :μ => Int.(basis.zlist.list) )

isadmissible(b, basis::Species1PBasisCtr) = (AtomicNumber(b.μ0) in basis.zlist.list)
isadmissible(b, basis::Species1PBasisNeig) = (AtomicNumber(b.μ) in basis.zlist.list)

# indexrange(basis::Species1PBasisCtr) = Dict( :μ0 => 1:length(basis.zlist) )
# indexrange(basis::Species1PBasisNeig) = Dict( :μ => 1:length(basis.zlist) )
