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

import os

from ase import Atoms
from ase.constraints import FixAtoms, StrainFilter
import ase.io
import ase.lattice.general_surface
from ase.optimize import BFGS
import numpy as np
import scipy.optimize

def fit_lattice_parameters_with_stress_tensor(
    cell, calculator, log_dir=None, verbose=True):
    """Optimizes the bulk lattice parameters using the stress tensor method.

    Requires a calculator that supports computing the stress tensor (e.g.,
    GPAW using a PW mode).
    """

    # Print header
    if verbose:
        print("BASC: FITTING LATTICE PARAMETERS (STRESS TENSOR METHOD)")
        print("Compound: %s" % cell.get_name())
        print("calculator: %s" % str(calculator))
        print("log_dir: %s" % log_dir)

    # Make log dir
    try: os.mkdir(log_dir)
    except OSError: pass

    # Set up optimizer
    relaxed_cell = cell.copy()
    relaxed_cell.set_calculator(calculator)
    sf = StrainFilter(relaxed_cell)
    dyn = BFGS(sf)

    # Run the optimizer
    dyn.run(fmax=0.005)

    # Save and return
    if verbose:
        ase.io.write("%s/lattice_final.cif" % log_dir, relaxed_cell)
    return relaxed_cell

def fit_lattice_parameters_with_optimizer(
    cell, calculator, orthorhombic=True, log_dir=None, verbose=True):
    """Optimizes the bulk lattice by minimizing the potential energy.

    Uses SciPy to solve the optimization problem of fitting the unit cell
    parameters while keeping the fractional atomic positions constant.  Uses
    method L-BFGS-B with bounds of -5% and +5% from the original values.

    -- orthorhombic (default True): fit just the `a` and `c` unit cell
       parameters.  If False, `a` and `b` will be treated independently.
    """

    # Print header
    if verbose:
        print("BASC: FITTING LATTICE PARAMETERS (OPTIMIZER METHOD)")
        print("Compound: %s" % cell.get_name())
        print("calculator: %s" % str(calculator))
        print("log_dir: %s" % log_dir)

    # Make log dir
    try: os.mkdir(log_dir)
    except OSError: pass

    # Set up optimizer
    relaxed_cell = cell.copy()
    relaxed_cell.set_calculator(calculator)
    x0 = np.diag(cell.cell)
    if orthorhombic: x0 = x0[0::2]  # remove the second element of the array
    bounds = [tuple(x) for x in np.array([x0*0.95, x0*1.05]).T]
    def optim_fn(x):
        if orthorhombic:
            relaxed_cell.set_cell([x[0], x[0], x[1]], scale_atoms=True)
        else:
            relaxed_cell.set_cell(x, scale_atoms=True)
        energy = relaxed_cell.get_potential_energy()
        if verbose:
            print(">> eval'd (%s): %.6f" % (str(x), energy))
        return energy

    # Run the optimization
    result = scipy.optimize.minimize(optim_fn, x0, method="L-BFGS-B", options={
        "eps": 1e-5,
        "gtol": 0.005
    }, bounds=bounds)

    # Save and return
    optim_fn(result.x)  # reset lattice parameters to solution
    if verbose:
        print(result)
        ase.io.write("%s/lattice_final.cif" % log_dir, relaxed_cell)
    return relaxed_cell

def relax_surface_cell(cell, calculator, log_dir=None, verbose=True,
                       hkl=(0,0,1), width=1,
                       fluid_layers=1, fixed_layers=1):
    """Make an empty surface and relax it."""

    # Print header
    if verbose:
        print("BASC: RELAXING EMPTY SURFACE CELL")
        print("Compound: %s" % cell.get_name())
        print("calculator: %s" % str(calculator))
        print("log_dir: %s" % log_dir)
        print("hkl: %s" % str(hkl))
        print("width: %d" % width)
        print("fluid_layers: %d" % fluid_layers)
        print("fixed_layers: %d" % fixed_layers)

    # Set up system
    total_layers = fluid_layers + fixed_layers
    surf = ase.lattice.general_surface.surface(
        cell.repeat((width,width,1)),
        hkl,
        total_layers,
        15.0)
    number_of_fixed_atoms = len(surf) * fixed_layers / total_layers
    c = FixAtoms(
        indices=list(range(number_of_fixed_atoms)))
    surf.set_constraint(c)
    surf.set_pbc((True,True,True))

    # Run the relaxation
    perform_relaxation(surf, calculator, "relax_surface_cell",
                       log_dir, verbose)

def relax_basc_result(surf, relaxed_surf, calculator,
                      fixed_layers=1, fluid_layers=1,
                      log_dir=None, verbose=True):
    """Relax the structure of a BASC output."""

    # Print header
    if verbose:
        print("BASC: RELAXING RESULT")
        print("Compound: %s" % relaxed_surf.get_name())
        print("Compound with Adsorbate: %s" % surf.get_name())
        print("calculator: %s" % str(calculator))
        print("log_dir: %s" % log_dir)
        print("fluid_layers: %d" % fluid_layers)
        print("fixed_layers: %d" % fixed_layers)

    # Make sure that the atoms are sorted by z coordinate (don't assume it).
    # This method of sorting the atoms feels a bit like a hack.  It has the
    # potential to break if ASE changes the way Atoms objects work.  "arrays"
    # is the property where atom info is stored.  We can't use "temp" directly
    # because the unit cell, PBC, and other properties are not copied over
    # from the original object.
    temp = Atoms(sorted(surf, key=lambda atom: atom.z))
    surf.arrays = temp.arrays

    # Freeze bottom layers.  Use relaxed_surf to count so that the adsorbate
    # atoms are not included in the number of surface atoms.
    total_layers = fluid_layers + fixed_layers
    number_of_fixed_atoms = len(relaxed_surf) * fixed_layers // total_layers
    c = FixAtoms(indices=list(range(number_of_fixed_atoms)))
    surf.set_constraint(c)

    return perform_relaxation(surf, calculator, "relax_best",
                              log_dir, verbose)

def perform_relaxation(surf, calculator, name, log_dir=None, verbose=True):
    """Run a relaxation procedure using BFGS and error recovery."""

    if log_dir is None:
        raise ValueError("Please specify a log_dir!")

    # Perform the relaxation
    n = 0
    while True:
        n += 1
        job_prefix = "%s/%s_%d" % (log_dir, name, n)
        try: os.mkdir(job_prefix)
        except OSError: pass
        traj_path = "%s/optim.traj" % job_prefix
        bak_path = "%s/backup.pkl" % job_prefix

        if verbose:
            # Export images and structure files for the surface.
            ase.io.write("%s/initial.eps" % job_prefix, surf,
                show_unit_cell=1)
            ase.io.write("%s/initial.cif" % job_prefix, surf)
            ase.io.write("%s/initial.xsf" % job_prefix, surf)

        surf.set_calculator(calculator)

        try:
            # Attempt to run the optimization procedure
            dyn = BFGS(surf, trajectory=traj_path, restart=bak_path)
            dyn.run(fmax=0.05, steps=50)

        except np.linalg.linalg.LinAlgError:
            # We encountered a linalg error, probably "Eigenvalues did not
            # converge". We can restart BFGS at the previous frame.
            if verbose:
                print("Encountered LinAlgError; restarting from "
                      "previous BFGS frame")
            surf = ase.io.read(traj_path)

        else:
            # All done
            break

    # Save the final structure
    if verbose:
        ase.io.write("%s/%s_final.cif" % (log_dir, name), surf)
    return surf

