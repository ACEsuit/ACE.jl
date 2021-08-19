# ACE.jl

<!-- [![tests](https://github.com/ACEsuit/ACE.jl/actions/workflows/Tests.yml/badge.svg)](https://github.com/ACEsuit/ACE.jl/actions/workflows/Tests.yml) -->
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/ACEsuit/ACE.jl/Tests) [![docs-dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://acesuit.github.io/ACE.jl/dev/) 
<!-- [![docs-stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://libatoms.github.io/ExtXYZ.jl/stable) -->
<!-- [![docs-dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://github.com/ACEsuit/ACE.jl/dev) -->


<!-- [![Build Status](https://travis-ci.com/JuliaMolSim/ACE.jl.svg?branch=master)](https://travis-ci.com/JuliaMolSim/ACE.jl) -->

<!-- [![Codecov](https://codecov.io/gh/JuliaMolSim/ACE.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaMolSim/ACE.jl) -->

Notes: 
* This is currently a development branch of ACE (though we are still tagging versions regularly). For the latest stable version see [DEV-v0.8.x](https://github.com/ACEsuit/ACE.jl/tree/dev-v0.8.x)
* [Preliminary Documentation](https://acesuit.github.io/ACE.jl/dev/), WIP.

This package implements approximation schemes for permutation and isometry invariant functions, with focus on modelling atomic interactions. It provides constructions of symmetric polynomial bases, such as permutation and isometry invariance.
Heavy use is made of trigonometric polynomials and spherical harmonics to obtain rotation invariance. Implementations of other symmetries are in preparation. 

* Main new development towards nonlinear models and AD is on `main` branch 
* AD is still a WIP, but expect this to be ready soon
* All previous branches are now in maintenance mode.

See [Documentation](https://acesuit.github.io/ACE.jl/dev/) for more details on ACE and for references. 


## License

`ACE.jl` is © 2019, Christoph Ortner

`ACE.jl` is published and distributed under the [Academic Software License v1.0 (ASL).](ASL.md)

`ACE.jl` is distributed in the hope that it will be useful for non-commercial academic research, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the ASL for more details.

You should have received a copy of the ASL along with this program; if not, write to Christoph Ortner, christophortner0@gmail.com. It is also published at [https://github.com/gabor1/ASL/blob/main/ASL.md](https://github.com/gabor1/ASL/blob/main/ASL.md).

You may contact the original licensor at `christophortner0@gmail.com`.
