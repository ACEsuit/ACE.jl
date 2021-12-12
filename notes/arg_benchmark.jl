#
# This script documents the cost of mapping from a structure 
# to an environment, which doubles the amount of memory used 
# (but this can of course be allocated), so primarily it is 
# the cost of copying the memory.
#
using ACE, StaticArrays, BenchmarkTools
using ACE: State

randstate() = 
           ( rr = randn(SVector{3, Float64}), 
             μ = rand([1,2,3,4,5]), 
             q = randn(), 
             p = randn(SVector{3, Float64}) )

function struct2env(Xs, i, J)
   xi = Xs[i]
   return [ ( xj = Xs[j]; 
                   (rr = xj.rr - xi.rr, 
                    μi = xi.μ, μj = xj.μ, 
                    qi = xi.q, qj = xj.q, 
                    pi = xi.p, pj = xj.p) )  for j in J ]
end
            
function struct2env!(Xs_i, Xs, i, J)
   xi = Xs[i]
   for (ij, j) in enumerate(J)
      xj = Xs[j]
      Xs_i[ij] =      (rr = xj.rr - xi.rr, 
                       μi = xi.μ, μj = xj.μ, 
                       qi = xi.q, qj = xj.q, 
                       pi = xi.p, pj = xj.p )
   end
   return Xs_i 
end

Xs = [ randstate() for _ = 1:1000 ]
i = rand(1:1000)
J = setdiff(unique(rand(1:1000, 35)), [i])

Xs_i = struct2env(Xs, i, J)
struct2env!(Xs_i, Xs, i, J)

@btime struct2env($Xs, $i, $J)
@btime struct2env!($Xs_i, $Xs, $i, $J)


##

struct Wrapper{SYMS, TT}
   x::NamedTuple{SYMS, TT}
end
randnt() = Wrapper((a = randn(), b = rand(), c = rand(1:5), 
                   d = randn(SVector{3, Float64})))
X = randnt()
@allocated randnt()
isbits(X)
isbits(X.x)


struct Wrapper1{NT} 
   x::NT
end
randnt1() = Wrapper1((a = randn(), b = rand(), c = rand(1:5), d = randn(SVector{3, Float64})))
X1 = randnt1()
@allocated randnt1()
isbits(X1)
isbits(X1.x)

##
