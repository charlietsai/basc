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

from ase import Atoms
import ase.io
from gpaw import mpi
import numpy as np
import pickle

from basc.basc import BASC

# Constants
log_dir = "logs/demo"
write_logs = (mpi.world.rank==0)
molecule_CO = Atoms("CO", positions=[(0.,0.,1.128),(0.,0.,0.)])

# Load the relaxed surface from Step 1.
relaxed_surf = ase.io.read("samples/Fe2O3_relaxed_surface.cif")

# Make the BASC instance.  Refer to the BASC docstrings for more
# options for the parameters.
basc = BASC(relaxed_surf, molecule_CO, 0, noise_variance=1e-4,
	        write_logs=write_logs)

# For the purposes of fitting the length scales, assume that the minimum is
# two standard deviations below the mean.
variance_transform=lambda v,Y: np.square((np.mean(Y)-np.min(Y))/2.)

# Fit the length scales
basc.fit_lengthscales(
	n=800,  # number of points to use in data set for fitting
	write_logs=write_logs,
	variance_transform=variance_transform
)

# Save length scales to a pickle file
if write_logs:
	pickle.dump(
		basc.lengthscales,
		open("samples/Fe2O3_length_scales.pkl", "wb"))
