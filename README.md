# ACE.jl

[![tests](https://github.com/ACEsuit/ACE.jl/actions/workflows/Tests.yml/badge.svg)](https://github.com/ACEsuit/ACE.jl/actions/workflows/Tests.yml)

<!-- [![Build Status](https://travis-ci.com/JuliaMolSim/ACE.jl.svg?branch=master)](https://travis-ci.com/JuliaMolSim/ACE.jl) -->

[![Codecov](https://codecov.io/gh/JuliaMolSim/ACE.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaMolSim/ACE.jl)

[Preliminary Documentation](https://juliamolsim.github.io/ACE.jl/dev/)

This package implements approximation schemes for permutation and isometry invariant functions, with focus on modelling atomic interactions. It provides constructions of symmetric polynomial bases, imposing permutation and isometry invariance.
Heavy use is made of trigonometric polynomials and spherical harmonics to obtain rotation invariance. There are also implementations of pure permutation invariant bases and of bases with only cylindrical symmetries for bond energies.
Documentation is a work in progress; if you wish to use the code please contact the author.



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
