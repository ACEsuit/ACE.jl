import os
from ase.calculators.lammpsrun import LAMMPS
from ase.io import read, write
from ase.optimize import LBFGS
from ase.lattice.cubic import Diamond
import time

os.environ["ASE_LAMMPSRUN_COMMAND"]="/Users/Cas/gits/lammps-ace/build/lmp"

parameters = {'pair_style': 'pace',
              'pair_coeff': ['* * CrFeH_loworder.yace Fe']}

files = ["CrFeH_loworder.yace"]

calc1 = LAMMPS(parameters=parameters, files=files)

from ase.build import bulk

at = bulk("Fe") * (2,2,2)
print(at)
at.set_calculator(calc1)

print("HELLO")

t1 = time.time()
at.get_forces()
t2 = time.time()
print((t2 - t1)/len(at))

# import pyjulip

# calc2 = pyjulip.ACE("./Si_B6_N7_18_lap_3.0_rep.json")

# at = Diamond("Si", latticeconstant=5.44) * (10,10,10)
# at.rattle(stdev=0.1)
# at.set_calculator(calc2)

# t1 = time.time()
# at.get_forces()
# t2 = time.time()
# (t2 - t1)/len(at)

# at.get_potential_energy()


# at.get_forces()

# dyn = LBFGS(at, trajectory='./pyjulip_relax.traj')
# dyn.run()
# #
# # al = read("./RSS_configs/pyjulip_relax.traj", ":")
# # write("./pyjulip_relax.xyz", al)
