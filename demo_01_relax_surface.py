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
This scripts is a demonstration of step 1 of the BASC framework, relaxing the
empty surface cell.  It must be performed before any other steps.

This step can benefit from multiple cores.

[Step 1] --> [Step 2] --> [Step 4]
         +-> [Step 3] -^
"""

try: from configparser import ConfigParser  # Python 3
except ImportError: from ConfigParser import ConfigParser  # Python 2

import os

import ase.io
import ase.parallel
from gpaw import GPAW, mpi

import basc.utils

config = ConfigParser()
config.read("BASC.ini")

# Load config options
log_dir = config.get("General", "log_dir")
results_dir = config.get("General", "results_dir")
cell = ase.io.read(config.get("Material", "cell"))
hkl = eval(config.get("Material", "hkl"))  # eval() to get the tuple
width = int(config.get("Material", "width"))
fluid_layers = int(config.get("Material", "fluid_layers"))
fixed_layers = int(config.get("Material", "fixed_layers"))
gpaw_kwargs = eval(config.get("GPAW", "kwargs"))

# Write logs iff we are the master process
write_logs = (mpi.world.rank==0)

# Make the log and results directories if necessary
try: os.mkdir(log_dir)
except OSError: pass
try: os.mkdir(results_dir)
except OSError: pass

# Run relaxation
relaxed_surf = basc.utils.relax_surface_cell(
    cell,  # unit cell of crystal
    GPAW(
        communicator=mpi.world.new_communicator(list(range(mpi.world.size))),
        txt=ase.parallel.paropen("%s/gpaw.log" % log_dir, "a"),
        **gpaw_kwargs
    ),
    log_dir,
    write_logs,
    hkl=hkl,
    width=width,
    fluid_layers=fluid_layers,
    fixed_layers=fixed_layers
)

# Save relaxed surface for next step
ase.io.write("%s/relaxed_surface.cif" % results_dir, relaxed_surf)
