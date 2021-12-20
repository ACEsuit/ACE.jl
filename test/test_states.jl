
module St
   abstract type AbstractState end 

   include("../src/states.jl")
end

using StaticArrays, BenchmarkTools, Test, LinearAlgebra

##


X = St.State( rr = randn(SVector{3, Float64}) )

SYMS = St._syms(X)
println(@test SYMS == (:rr,))

TT = St._tt(X)
println(@test( TT == Tuple{SVector{3, Float64}} ))

TDX = St.dstate_type(X)
println(@test TDX == St.DState{NamedTuple{(:rr,), Tuple{SVector{3, Float64}}}})


dX = St.DState(X)
println(@test typeof(dX) == St.dstate_type(X))
println(@test dX.rr == X.rr)


cTDX = St.dstate_type(0.0im, X)
println(@test cTDX == St.DState{NamedTuple{(:rr,), Tuple{SVector{3, ComplexF64}}}})

cdX = complex(dX)
println(@test( cdX == St.DState(rr = X.rr .+ 0im) ))
println(@test( real(cdX) == dX ))
println(@test( imag(cdX) == St.DState(rr = zero(SVector{3, Float64})) ))

# not sure how to test this, but at least it should work:
@show rand(TDX)
@show randn(TDX)
@show zero(TDX)

@show rand(X)
@show randn(X)
@show zero(X)

@info("arithmetic")
println(@test( X + dX == St.State(rr = X.rr + dX.rr) ))
println(@test( X - dX == St.State(rr = X.rr - dX.rr) ))
println(@test( dX + cdX == St.DState(rr = dX.rr + cdX.rr) ))
println(@test( dX - cdX == St.DState(rr = dX.rr - cdX.rr) ))

a = randn() 
println(@test( a * dX == St.DState(rr = a * dX.rr) ))
println(@test( dX * a == St.DState(rr = a * dX.rr) ))
println(@test( -dX == St.DState(rr = -dX.rr) ))

println(@test( dot(dX, cdX) == dot(dX.rr, cdX.rr) ))
println(@test( dot(cdX, dX) == dot(cdX.rr, dX.rr) ))

dX1, dX2 = randn(cdX), randn(cdX)
println(@test St.contract(dX1, dX2) == sum(dX1.rr .* dX2.rr) )
println(@test St.contract(dX1, dX2) != dot(dX1, dX2) )

println(@test isapprox(dX, cdX))

println(@test norm(dX) == norm(dX.rr) )
println(@test St.sumsq(dX) == St.sumsq(dX.rr) )
println(@test St.normsq(dX) == St.normsq(dX.rr) )
