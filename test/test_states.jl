

using ACE, StaticArrays, BenchmarkTools, Test, LinearAlgebra
using ACE.Testing 

using ACE: State, DState

##

X = State( rr = randn(SVector{3, Float64}) )

SYMS = ACE._syms(X)
println(@test SYMS == (:rr,))

TT = ACE._tt(X)
println(@test( TT == Tuple{SVector{3, Float64}} ))

TDX = ACE.dstate_type(X)
println(@test TDX == DState{NamedTuple{(:rr,), Tuple{SVector{3, Float64}}}})


dX = DState(X)
println(@test typeof(dX) == ACE.dstate_type(X))
println(@test dX.rr == X.rr)


cTDX = ACE.dstate_type(0.0im, X)
println(@test cTDX == DState{NamedTuple{(:rr,), Tuple{SVector{3, ComplexF64}}}})

cdX = complex(dX)
println(@test( cdX == DState(rr = X.rr .+ 0im) ))
println(@test( real(cdX) == dX ))
println(@test( imag(cdX) == DState(rr = zero(SVector{3, Float64})) ))

# not sure how to test this, but at least it should work:
@show rand(TDX)
@show randn(TDX)
@show zero(TDX)

@show rand(X)
@show randn(X)
@show zero(X)

@info("arithmetic")
println(@test( X + dX == State(rr = X.rr + dX.rr) ))
println(@test( X - dX == State(rr = X.rr - dX.rr) ))
println(@test( dX + cdX == DState(rr = dX.rr + cdX.rr) ))
println(@test( dX - cdX == DState(rr = dX.rr - cdX.rr) ))

a = randn() 
println(@test( a * dX == DState(rr = a * dX.rr) ))
println(@test( dX * a == DState(rr = a * dX.rr) ))
println(@test( -dX == DState(rr = -dX.rr) ))

println(@test( dot(dX, cdX) == dot(dX.rr, cdX.rr) ))
println(@test( dot(cdX, dX) == dot(cdX.rr, dX.rr) ))

dX1, dX2 = randn(cdX), randn(cdX)
println(@test ACE.contract(dX1, dX2) == sum(dX1.rr .* dX2.rr) )
println(@test ACE.contract(dX1, dX2) != dot(dX1, dX2) )

println(@test isapprox(dX, cdX))

println(@test norm(dX) == norm(dX.rr) )
println(@test ACE.sumsq(dX) == ACE.sumsq(dX.rr) )
println(@test ACE.normsq(dX) == ACE.normsq(dX.rr) )

##

@info("performance/allocation test ")

function bm_copy!(Y, X, a)
   for i = 1:length(Y)
      Y[i] = X[i] * a 
   end 
   return Y
end

Xs = [ rand(PositionState{Float64}) for _=1:100 ]
Ys = [ zero(PositionState{Float64}) for _=1:100 ]
a = rand() 
bm = @benchmark bm_copy!($Ys, $Xs, $a)
display(bm) 
println(@test bm.allocs == 0 && bm.memory == 0)
