
# Introduction

This package implements approximation schemes for permutation and isometry invariant functions, with focus on modelling atomic interactions. It provides constructions of symmetric polynomial bases, imposing permutation and isometry invariance. Heavy use is made of trigonometric polynomials and spherical harmonics to obtain rotation invariance.

The main scheme currently implemented is based on the Atomic Cluster Expansion (ACE) described in

   Drautz, R.: Atomic cluster expansion for accurate and transferable interatomic potentials. Phys. Rev. B Condens. Matter. 99, 014104 (2019). doi:10.1103/PhysRevB.99.014104

A more detailed description and variations are discussed in

      M. Bachmayr, G. Csanyi, G. Dusson, S. Etter, C. van der Oord, and C. Ortner. Approximation of potential energy surfaces with spherical harmonics. arXiv:1911.03550v2; [http](https://arxiv.org/abs/1911.03550) [PDF](https://arxiv.org/pdf/1911.03550.pdf)

There are also implementations of pure permutation invariant bases and of bases with only cylindrical symmetries for bond energies.
