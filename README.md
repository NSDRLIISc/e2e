# e2e
End-to-end code for predicting Curie temp. and other magnetic properties of 2D materials. Re-optimied for Quantum Espresso (QE)

The code needs Python > 3.6 to run. The following libraries are required all of which can be installed using pip. These packages in turn have their own dependencies which would be installed automatically.

automatminer, ase, numpy, sympy, numba

pymatgen may not recognise a lot of elements as magnetic. you have to manually edit the pymatgent's default_magmoms.yaml and add the elements with a suitable starting MAGMOM.

The code name is e2e.py, and it shoud be run in a directory containing all the necessary input files (input, structure file and optionally input_MC). An example input and input_MC file is given below. The tags are self-explanatory.