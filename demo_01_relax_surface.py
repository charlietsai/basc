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
This scripts is a demonstration of step 1 of the BASC framework, relaxing the
empty surface cell.  It must be performed before any other steps.

This step can benefit from multiple cores.

[Step 1] --> [Step 2] --> [Step 4]
         +-> [Step 3] -^
"""

import os

import ase.io
import ase.parallel
from gpaw import GPAW, mpi

import basc.utils

# Constants
log_dir = "logs/demo"
write_logs = (mpi.world.rank==0)

# Load the unit cell.
# You could also load it from Materials Project, for example.
cell = ase.io.read("samples/Fe2O3_unit_cell.cif")

# Make the log directory if necessary
try: os.mkdir(log_dir)
except OSError: pass

# Run relaxation
relaxed_surf = basc.utils.relax_surface_cell(
    cell,  # unit cell of crystal
    GPAW(
        communicator=mpi.world.new_communicator(list(range(mpi.world.size))),
        txt=ase.parallel.paropen("%s/gpaw.log" % log_dir, "a"),
        spinpol=True
    ),
    log_dir,
    write_logs,
    hkl=(0,0,1),  # Miller indices
    width=1,  # number of times to copy unit cell in x/y
    fluid_layers=1,  # number of layers in which atoms can move
    fixed_layers=1  # number of layers in which atoms are fixed
)

# Save relaxed surface for next step
ase.io.write("samples/Fe2O3_relaxed_surface.cif", relaxed_surf)
