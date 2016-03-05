#!/usr/bin/env python

# Copyright (c) 2016, Shane Frederic F. Carr
# 
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
# 
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

"""
This scripts is a demonstration of step 4 of the BASC framework, performing
the actual Bayesian optimization.

This step can benefit from multiple cores.

[Step 1] --> [Step 2] --> [Step 4]
         +-> [Step 3] -^
"""

import pickle

from ase import Atoms
import ase.io
import ase.parallel
from gpaw import mpi
import numpy as np

from basc.basc import BASC
from basc.gpaw_trained import GPAWTrained

# Constants
log_dir = "logs/demo"
write_logs = (mpi.world.rank==0)
molecule_CO = Atoms("CO", positions=[(0.,0.,1.128),(0.,0.,0.)])

# Load the relaxed surface from Step 1.
relaxed_surf = ase.io.read("samples/Fe2O3_relaxed_surface.cif")

# Make the BASC instance.
basc = BASC(relaxed_surf, molecule_CO, 0, noise_variance=1e-4,
            write_logs=write_logs)

# Load the length scales from Step 2.
basc.lengthscales = pickle.load(
    open("samples/Fe2O3_length_scales.pkl", "rb"))

# Load the traces from Step 3.
training_Y = np.load("samples/Fe2O3_training.npz")["Y"]

# Make the GPAWTrained instance
calculator = GPAWTrained(
    training_Y,
    communicator=mpi.world.new_communicator(list(range(mpi.world.size))),
    txt=ase.parallel.paropen("%s/gpaw.log" % log_dir, "a"),
    spinpol=True
)
basc.set_calculator(calculator)

# Print header
if write_logs:
    print("RUNNING BAYES OPT")
    print("log_dir: %s" % log_dir)

# Run BO iterations
for i in range(200):
    basc.run_iteration(write_logs=write_logs)

# Save the best result
bestX,bestY,bestIter,bestAtoms = basc.best()
print("Best Result: iteration %d, y=%f, x=%s" % (bestIter,bestY,str(bestX)))
ase.io.write("samples/Fe2O3_best.cif", bestAtoms)
