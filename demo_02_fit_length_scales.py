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
This scripts is a demonstration of step 2 of the BASC framework, fitting the
length scales of the GP.  It may be performed in parallel with step 3.

THIS STEP DOES NOT CURRENTLY BENEFIT FROM MULTIPLE CORES.

[Step 1] --> [Step 2] --> [Step 4]
         +-> [Step 3] -^
"""

try: from configparser import ConfigParser  # Python 3
except ImportError: from ConfigParser import ConfigParser  # Python 2

import pickle

from ase import Atoms
import ase.io
from gpaw import mpi
import numpy as np

from basc.basc import BASC

config = ConfigParser()
config.read("BASC.ini")

# Load config options
log_dir = config.get("General", "log_dir")
results_dir = config.get("General", "results_dir")
seed = int(config.get("General", "seed"))
adsorbate = ase.io.read(config.get("Adsorbate", "structure"))
phi_length = int(config.get("Adsorbate", "phi_length"))

# Write logs iff we are the master process
write_logs = (mpi.world.rank==0)

# Load the relaxed surface from Step 1.
relaxed_surf = ase.io.read("%s/relaxed_surface.cif" % results_dir)

# Make the BASC instance.  Refer to the BASC docstrings for more
# options for the parameters.
basc = BASC(relaxed_surf, adsorbate, phi_length, noise_variance=1e-4,
	          seed=seed, write_logs=write_logs)

# Fit the length scales
basc.fit_lengthscales(
	n=800,  # number of points to use in data set for fitting
	write_logs=write_logs
)

# Save length scales to a pickle file
if write_logs:
	pickle.dump(
		basc.lengthscales,
		open("%s/length_scales.pkl" % results_dir, "wb"))
