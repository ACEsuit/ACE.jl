
using ACE, StaticArrays

𝒓 = EuclideanVectorState("𝒓")
μ = SpeciesState("μ")
X1 = μ ⊗ 𝒓
X = AtomState()

X1.rr
X1.mu

@code_llvm X.rr
@code_llvm X1.rr

length(X)  # wait and see what we want `length` to mean...

using BenchmarkTools

function test_perf(X)
   rr = X.rr + rand(SVector{3, Float64})
   return rr
end

@code_warntype test_perf(X1)
@code_warntype getproperty(X1, :rr)

struct S3{T}
   fields::T
end

Base.getproperty(x::S3, sym::Symbol) =
      sym === :fields ? getfield(x, :fields) : getproperty(x.fields, sym)

X3 = S3(X2)

X2 = (mu = AtomicNumber(0), rr = zero(SVector{3, Float64}))


@code_warntype test_perf(X3)

#----

using StaticArrays

struct S{T}
   fields::T
end

Base.getproperty(x::S, sym::Symbol) =
      sym === :fields ? getfield(x, :fields) : getindex(x.fields, sym)

T = SVector{3, Float64}
X1 = (a = 0, b = zero(T))
X2 = S(X1)

function test(X, T)
   rand(T) + X.b
end

@code_warntype test(X1, T)

@code_warntype test(X2, T)
