using SHIPs
using Test

@testset "SHIPs.jl" begin
    include("test_jacobi.jl")
    include("test_ylm.jl")
    include("test_cg.jl")
end
