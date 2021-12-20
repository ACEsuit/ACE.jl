
module St
   abstract type AbstractState end 

   include("../src/states.jl")
end

using StaticArrays, BenchmarkTools, Test 

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

