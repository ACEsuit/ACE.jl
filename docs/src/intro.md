
# Introduction

This package implements approximation schemes for permutation and isometry equi-variant functions (including invariant scalars, equi-variant vectors and tensors). Although the original focus was on modelling atomic interactions, the scope is in principle much broader hence the `ACE.jl` core library is agnostic about the application domain. It provides constructions of symmetric polynomial bases, imposing permutation and isometry invariance. Heavy use is made of trigonometric polynomials and spherical harmonics to impose the symmetries.

The implemention is based on the Atomic Cluster Expansion (ACE) described in the following references:

* Drautz, R.: Atomic cluster expansion for accurate and transferable interatomic potentials. Phys. Rev. B Condens. Matter. 99, 014104 (2019). doi:10.1103/PhysRevB.99.014104
* M. Bachmayr, G. Csanyi, G. Dusson, S. Etter, C. van der Oord, and C. Ortner. Atomic cluster expansion: Cluster Expansion: Completeness, Efficiency and Stability. arXiv:1911.03550v3; [[http]](https://arxiv.org/abs/1911.03550) [[PDF]](https://arxiv.org/pdf/1911.03550.pdf)
* Drautz, R.: Atomic cluster expansion of scalar, vectorial, and tensorial properties including magnetism and charge transfer, Phys. Rev. B 102, 024104, 2020 [[http]](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.102.024104)
