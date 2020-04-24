

import JuLIP.Potentials: z2i

import JuLIP: alloc_temp, alloc_temp_d,
              cutoff,
              evaluate, evaluate_d,
              evaluate!, evaluate_d!,
              SitePotential

import JuLIP.MLIPs: IPBasis, alloc_B, alloc_dB

import Base: Dict, convert, ==

# prototypes for space transforms and cutoffs
function transform end
function transform_d end
function fcut end
function fcut_d end


abstract type RadialBasis end

abstract type OneParticleBasis end


# Some methods for generating random samples
function rand_radial end

function rand_sphere()
   R = randn(JVecF)
   return R / norm(R)
end

rand_vec(J::RadialBasis) = rand_radial(J) *  rand_sphere()
rand_vec(J::RadialBasis, N::Integer) = [ rand_vec(J) for _ = 1:N ]
