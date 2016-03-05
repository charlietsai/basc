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
This scripts is a demonstration of step 3 of the BASC framework, obtaining
training data.  It may be performed in parallel with step 2.  Step 3 may be
omitted if you intend to use a calculator other than GPAWTrained.

This step can benefit from multiple cores.  To enhance the speed even more,
one can modify this code to assign different training points to different
subsets of the cores.

[Step 1] --> [Step 2] --> [Step 4]
         +-> [Step 3] -^
"""

from ase import Atoms
import ase.io
import ase.parallel
from gpaw import GPAW, mpi
import numpy as np

from basc.basc import BASC
from basc.gpaw_trained import obtain_traces

# Constants
log_dir = "logs/demo"
write_logs = (mpi.world.rank==0)
molecule_CO = Atoms("CO", positions=[(0.,0.,1.128),(0.,0.,0.)])

# Load the relaxed surface from Step 1.
relaxed_surf = ase.io.read("samples/Fe2O3_relaxed_surface.cif")

# Make the BASC instance.  Refer to the BASC docstrings for more
# options for the parameters.
basc = BASC(relaxed_surf, molecule_CO, 0, noise_variance=1e-6,
            write_logs=write_logs)

# Obtain training points for GPAWTrained
training_X = basc.sobol_points(8)
training_atomses = [basc.atoms_from_point(point) for point in training_X]
training_Y = obtain_traces(
    training_atomses,
    write_logs=write_logs,
    communicator=mpi.world.new_communicator(list(range(mpi.world.size))),
    txt=ase.parallel.paropen("%s/gpaw.log" % log_dir, "a"),
    spinpol=True
)

# Save the traces
np.savez("samples/Fe2O3_training.npz", X=training_X, Y=training_Y)
