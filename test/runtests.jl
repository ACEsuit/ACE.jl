using SHIPs
using Test, Printf, LinearAlgebra, StaticArrays, BenchmarkTools, Test

print_tf(::Test.Pass) = printstyled("✓", bold=true, color=:green)
print_tf(::Test.Fail) = printstyled("x", bold=true, color=:red)

@testset "SHIPs.jl" begin
    include("test_jacobi.jl")
    include("test_transforms.jl")
    include("test_ylm.jl")
    include("test_cg.jl")
    include("test_basis.jl")
end
