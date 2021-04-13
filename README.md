# ACE.jl

[![tests](https://github.com/ACEsuit/ACE.jl/actions/workflows/Tests.yml/badge.svg)](https://github.com/ACEsuit/ACE.jl/actions/workflows/Tests.yml)

<!-- [![Build Status](https://travis-ci.com/JuliaMolSim/ACE.jl.svg?branch=master)](https://travis-ci.com/JuliaMolSim/ACE.jl) -->

<!-- [![Codecov](https://codecov.io/gh/JuliaMolSim/ACE.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaMolSim/ACE.jl) -->

[Preliminary Documentation](https://juliamolsim.github.io/ACE.jl/dev/)

This package implements approximation schemes for permutation and isometry invariant functions, with focus on modelling atomic interactions. It provides constructions of symmetric polynomial bases, imposing permutation and isometry invariance.
Heavy use is made of trigonometric polynomials and spherical harmonics to obtain rotation invariance. There are also implementations of pure permutation invariant bases and of bases with only cylindrical symmetries for bond energies.
Documentation is a work in progress; if you wish to use the code please contact the author.

## Usage

The ACE basis can be set up using the following function `rpi_basis()` containing the species, correlation order `N`, polynomial degree `maxdeg`, nearest neighbour distance `r0` and inner/outer cutoff radii `rin` and `rcut`. Other parameters for defining the size of the basis (provided by `length(B)`) are `wL` and `csp`. 
```
B = rpi_basis(species = :Si,
      N = 3,                        # correlation order = body-order - 1
      maxdeg = 13,                  # polynomial degree
      r0 = r0,                      # estimate for NN distance
      D = SparsePSHDegree(; wL=1.3, csp=1.0),
      rin = 0.65*r0, rcut = 5.5,    # domain for radial basis (cf documentation)
      pin = 0)  
```
This basis can then be used in combination with `IPFitting.jl` to create a least squares system `dB` used for fitting. 
```
al = IPFitting.Data.read_xyz("./Si.xyz", energy_key="dft_energy", force_key="dft_force", virial_key="dft_virial")
dB = LsqDB("", B, al)
```
We can then fit the potential using `lsqfit()` given a set of `weights` and reference one body potential `Vref`.
```
weights = Dict("default" => Dict("E" => 15.0, "F" => 1.0 , "V" => 1.0 ))
Vref = OneBody(Dict("Si" => -158.54496821))
IP, lsqinfo = lsqfit(dB; weights = weights, Vref = Vref, asmerrs = true, solver=(:lap, 1.2));
```
Returning a fitted interatomic potential `IP` and an `lsqinfo` dictionary containing information of the fit, such as the errors which can be displayed in table as follows.
```
rmse_table(lsqinfo["errors"])
```

## References

When using this software, please cite the following references

* Drautz, R.: Atomic cluster expansion for accurate and transferable interatomic potentials. Phys. Rev. B Condens. Matter. 99, 014104 (2019). doi:10.1103/PhysRevB.99.014104

* M. Bachmayr, G. Csanyi, G. Dusson, S. Etter, C. van der Oord, and C. Ortner. Approximation of potential energy surfaces with spherical harmonics. arXiv:1911.03550v2; [http](https://arxiv.org/abs/1911.03550) [PDF](https://arxiv.org/pdf/1911.03550.pdf)


## License

`ACE.jl` is © 2019, Christoph Ortner

`ACE.jl` is published and distributed under the [Academic Software License v1.0 (ASL).](ASL.md)

`ACE.jl` is distributed in the hope that it will be useful for non-commercial academic research, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the ASL for more details.

You should have received a copy of the ASL along with this program; if not, write to Christoph Ortner, christophortner0@gmail.com. It is also published at [https://github.com/gabor1/ASL/blob/main/ASL.md](https://github.com/gabor1/ASL/blob/main/ASL.md).

You may contact the original licensor at `christophortner0@gmail.com`.
