
# ACE.jl Documentation

This package implements approximation schemes for symmetric functions (including invariant scalars, equi-variant vectors and tensors ...). The origin of this effort was in modelling atomic interactions, hence the most complete implementation focus on permutations and isometries, but much of the code is more general. Extensions to other symmetries are therefore planned for the near future. 

Although the original focus was on modelling atomic interactions, the scope is in principle much broader hence the `ACE.jl` core library is agnostic about the application domain. It provides constructions of symmetric polynomial bases. Application-specific layers are provided in the related packaged within [ACEsuit](https://github.com/ACEsuit).

The original implemention was based on the Atomic Cluster Expansion (ACE) described in the following references:

* Drautz, R.: Atomic cluster expansion for accurate and transferable interatomic potentials. Phys. Rev. B Condens. Matter. 99, 014104 (2019). doi:10.1103/PhysRevB.99.014104, [[html]](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.99.014104)
* M. Bachmayr, G. Csanyi, G. Dusson, S. Etter, C. van der Oord, and C. Ortner. Atomic cluster expansion: Cluster Expansion: Completeness, Efficiency and Stability. arXiv:1911.03550v3; [[http]](https://arxiv.org/abs/1911.03550) [[PDF]](https://arxiv.org/pdf/1911.03550.pdf)
* Drautz, R.: Atomic cluster expansion of scalar, vectorial, and tensorial properties including magnetism and charge transfer, Phys. Rev. B 102, 024104, 2020 [[http]](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.102.024104)



```@contents
Pages = ["gettingstarted.md",
         "math.md", 
         "devel.md",
         "docs.md"]
Depth = 2
```

### Editing and Building the Documentation Locally

To build the documentation locally, use the `make.jl` script. Simply switch to `ACE/docs` and execute `julia --project=.. make.jl`, then open `./build/index.html`.
