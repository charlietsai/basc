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

# 156621, 156626, 156627
# 156751, 156752, 156753, 156754, 156755
# 156801, 156802, 156803, 156806, 156818, 156819
# 156851, 156852
# 156996, 156997, 156998
# 156999

"""
This scripts is a demonstration of step 4 of the BASC framework, performing
the actual Bayesian optimization.

This step can benefit from multiple cores.

[Step 1] --> [Step 2] --> [Step 4]
         +-> [Step 3] -^
"""

try: from configparser import ConfigParser  # Python 3
except ImportError: from ConfigParser import ConfigParser  # Python 2

import pickle

from ase import Atoms
import ase.io
import ase.parallel
from gpaw import mpi
import numpy as np

from basc.basc import BASC
from basc.gpaw_trained import GPAWTrained

config = ConfigParser()
config.read("BASC.ini")

# Load config options
log_dir = config.get("General", "log_dir")
results_dir = config.get("General", "results_dir")
seed = int(config.get("General", "seed"))
adsorbate = ase.io.read(config.get("Adsorbate", "structure"))
phi_length = int(config.get("Adsorbate", "phi_length"))
gpaw_kwargs = eval(config.get("GPAW", "kwargs"))
lengthscale_influence = float(config.get("Optimization", "lengthscale_influence"))
variance_ratio = float(config.get("Optimization", "variance_ratio"))

# Write logs iff we are the master process
write_logs = (mpi.world.rank==0)

# Load the relaxed surface from Step 1.
relaxed_surf = ase.io.read("%s/relaxed_surface.cif" % results_dir)

# Make the BASC instance.
basc = BASC(relaxed_surf, adsorbate, phi_length, noise_variance=1e-4,
            seed=seed, write_logs=write_logs)

# Load the length scales from Step 2.
basc.lengthscales = pickle.load(
    open("%s/length_scales.pkl" % results_dir, "rb"))
if write_logs:
    print("lengthscales: %s" % str(basc.lengthscales))

# Load the traces from Step 3.
training_Y = np.load("%s/training.npz" % results_dir)["Y"]

# Make the GPAWTrained instance
calculator = GPAWTrained(
    training_Y,
    communicator=mpi.world.new_communicator(list(range(mpi.world.size))),
    txt=ase.parallel.paropen("%s/gpaw.log" % log_dir, "a"),
    **gpaw_kwargs
)
basc.set_calculator(calculator)

# Calculate how many iterations to run.
# Set it to be twice the "halflife" of the decaying variance function.  Read
# the BASC auto_gp docstring for more details.
influence_frac = basc.observation_influence_fraction(lengthscale_influence)
num_bo_iter = int(2/influence_frac)

# Print header
if write_logs:
    print("RUNNING BAYES OPT")
    print("log_dir: %s" % log_dir)
    print("num_bo_iter: %d" % num_bo_iter)

# Run BO iterations
for i in range(200):
    basc.run_iteration(
        write_logs = write_logs,
        mean_function = lambda D: np.mean(training_Y[:,-1]),
        base_variance = np.std(training_Y[:,-1]) * variance_ratio,
        lengthscale_influence = lengthscale_influence
    )

# Save the best result
bestX,bestY,bestIter,bestAtoms = basc.best()
if write_logs:
    print("Best Result: iteration %d, y=%f, x=%s" % (bestIter,bestY,str(bestX)))
ase.io.write("%s/best.cif" % results_dir, bestAtoms)
