
# Getting Started

(this is an early draft of this document; please send me comments!)

## Installation

1. Install [Julia](https://julialang.org), the latest versions of `ACE.jl` likely requires `v1.5` or higher, at the moment we don't perform rigorous tests on this. 
2. Install the [`MolSim` registry](https://github.com/JuliaMolSim/MolSim); from the Julia REPL, switch to package manager `]` and then run
```julia
registry add https://github.com/JuliaMolSim/MolSim.git
```
3. Install some important registered packages; from Julia REPL / package manager:
```julia
add PyCall IJulia    # add more important packages from General registry
add ACE              # maybe add other packages from MolSim registry
```
Add other packages you think you might need, e.g. 
```
add PyCall IJulia
add JuLIP ASE ACEatoms    # for modelling with atoms and molecules
```
4. For fitting interatomic potenitials, you need to install also [`IPFitting.jl`](https://github.com/cortner/IPFitting.jl),
```julia
add IPFitting
```
(Keep fingers crossed and hope it will be compatible with the current version of `ACE.jl`...) Other kinds of models are currently not supported by `IPFitting.jl` but regression must be performed "manually". Providing a generic regression code for ACE models is high on our priority list.


## Workflow

For now, please see [ACEsuite website](https://acesuit.github.io) for some initial examples. 