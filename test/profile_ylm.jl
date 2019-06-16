
##  TODO: there is something wrong here! => test errors due to sqrt(neg numbers)
@info("Test 3: Old vs New Timing")
L = 15
θ = rand() * π
cos_θ = cos(θ)
φ = (rand()-0.5) * 2*π
r = 0.1+rand()
R = SVector(r*sin(θ)*cos(φ), r*sin(θ)*sin(φ), r*cos(θ))
x, y, z, s = SphericalHarmonics.compute_rxz(R)
coeff = SphericalHarmonics.compute_coefficients(L)
P = Vector{Float64}(undef, SphericalHarmonics.sizeP(L))
SphericalHarmonics.compute_p!(L, z, coeff, P)
Y = Vector{ComplexF64}(undef, SphericalHarmonics.sizeY(L))
@info("Old Implementation:")
@btime compute_y!($L, $cos_θ, $φ, $P, $Y)
@info("New Implementation:")
@btime cYlm_from_cart!($Y, $L, $r, $x, $z, $s, $P)
@info("Experimental Real Spherical Harmonics:")
Yr = Vector{Float64}(undef, SphericalHarmonics.sizeY(L))
@btime SphericalHarmonics.rYlm_from_cart!($Y, $L, $r, $x, $z, $s, $P)
