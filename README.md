Bayesian Active Site Calculator (BASC)
======================================

This repository contains the source code behind the Bayesian Active Site
Calculator (BASC).  BASC is a framework for rapidly and accurately searching
for the global minimum on potential energy surfaces.

## Dependencies

In order to run BASC, you need a working installation of:

1. [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/),
    available in the yum/apt-get package "python-ase"
2. [GPAW](https://wiki.fysik.dtu.dk/gpaw/),
    available in the yum/apt-get package "gpaw"
    (GPAW is not required if you don't intend to use the accelerated DFT
    calculator)
3. [GPy](https://github.com/SheffieldML/GPy),
    available in the pip package "gpy"
4. A Python runtime that supportes these packages (e.g. Python 2.7)

I recommend running BASC on Linux, although it should run on any platform that
supports these dependencies.

## Quick Start

To find the optimal binding site for a molecule on a surface using the fast
Lennard-Jones potential energy calculator (the default if no other calculator
is specified):

```python
import ase.io
from basc.basc import BASC

# Read in your pre-relaxed surface slab and molecule
surface = ase.io.read("my_surface.cif")
molecule = ase.io.read("my_molecule.xyz")

# Make the BASC instance
basc = BASC(surface, molecule)

# Run BASC with all the default settings
point, energy, iter, solution = basc.run()

# Print and save the solution
print("Solution (%.6f eV) found on iteration %d" % (energy, iter))
ase.io.write("solution.cif", solution)
```

Set `basc.calculator` to the ASE-compatible calculator interface you want
to use.  For BASC's accelerated GPAW calculator:

```python
from basc.gpaw_trained import GPAWTrained, obtain_traces

# Make the calculator instance
calculator = GPAWTrained()

# Populate the calculator instance with traces corresponding to several
# molecule-surface Atoms objects well-distributed over the space
calculator.obtain_traces(basc.sobol_atomses(8))

# Hand the calculator to BASC
basc.calculator = calculator
```

You can obtain traces in a separate pre-processing job and save them in a
numpy file:

```python
import numpy as np

# Pre-processing job: run obtain_traces and save them to a numpy file
calculator = GPAWTrained()
calculator.obtain_traces(basc.sobol_atomses(8))
np.save("traces.npy", calculator.traces)

# Main job: skip the time-intensive call to obtain_traces
calculator = GPAWTrained()
calculator.traces = np.load("traces.npy")
basc.calculator = calculator
```

GPAWTrained takes the same keyword arguments as vanilla GPAW.  For example:

```python
# Example calculator instance with keyword arguments
calculator = GPAWTrained(
    spinpol = True,
    xc = "PBE",
    kpts = {"density": 2.5, "even": True}
)
```

Here's how to set up BASC and GPAW to take advantage of multiple cores:

```python
from ase.parallel import paropen
from gpaw import mpi

# MPI communicator object for GPAW
communicator = mpi.world.new_communicator(list(range(mpi.world.size)))

# Tell BASC to be verbose on only one core, so that messages aren't logged
# multiple times!
basc = BASC(surface, molecule, verbose=(mpi.world.rank==0))

# Make the calculator instance (you can include other kwargs)
calculator = GPAWTrained(
    txt = paropen("gpaw.log"),
    communicator = communicator
)
```

Since BASC performs a global search without any structural optimizations, it
is good practice to perform one additional structural relaxation on BASC's
result if you need, for example, the energy of adsorption.

```python
import basc.relaxation
from gpaw import GPAW

calculator = GPAW()
basc.relaxation.relax_basc_result(solution, surface, calculator,
                                  log_dir="logs", verbose=True)

ase.io.write("solution_relaxed.cif", solution)
```

## Tips and Tricks

Read the docstrings for details on the full array of options available for
BASC and its subroutines.  This section outlines a few highlights.

### Linear and Axis-Symmetric Molecules

Molecules that are linear (like CO₂) and otherwise rotationally symmetric
(like CH₄) inherently have a much smaller space of configurations than large,
less-symmetric molecules.  For example, consider methane, CH₄.  If an axis
is extended through one of the C-H bonds, rotating the molecule by 120
degrees (2/3 π radians) along that axis will result in the same orientation
as 0 degrees.


You can tell BASC about this symmetry via the *phi_length* option:

```python
molecule = ase.io.read("CH4.xyz")
basc = BASC(surface, molecule,
    phi_length = np.pi*2/3
)
```

The name "phi_length" originates from the convention of Euler coordinates
used by ASE.  The first rotation occurs by rotating the molecule about the
*z* axis by *phi* radians.

Linear molecules should have `phi_length=0`.  This will cause BASC to use a
spherical kernel for the Euler coordinates instead of three regular kernels.
For more detail on the spherical kernel, refer to BASC's academic article.

> **IMPORTANT:** You should always specify the phi_length when appropriate!
> Not doing so will unnecessarilly slow down the optimization procedure and
> may result in poor convergence.

> **CAUTION:** When specifying phi_length, make sure that your molecule is
> oriented such that the axis of symmetry is the *z* axis!  If it's not,
> you will get undefined results!


### Placing Molecule on Surface

By default, BASC will search over all *x* and *y* coordinates of the surface
cell.  However, you can (and should) tell BASC to only explore a smaller
region of the cell if your cell has multiple periodic regions.

You can specify the boundaries for your periodic region as follows:

```python
basc = BASC(surface, molecule,
    xbounds = (0, 0.5),
    ybounds = (0, 0.5)
)
```

The above example will only place the atom in the bottom-left quadrant of the
surface cell.

BASC also has built-in support for "root-2" unit cells.  You can tell BASC
that your cell is root-2 as follows:

```python
import basc.utils

basc = BASC(surface, molecule,
    add_adsorbate_method = basc.utils.add_adsorbate_root2
)
```

> **CAUTION:** Ensure that the region you specify is periodic!  This means
> the energy along the left boundary should be the same as along the right
> boundary, and likewise for the top and bottom boundaries.

You can also specify the range of *z* values over which your molecule is
placed on the surface.  It should cover a wide enough range such that the
molecule is sufficiently close to the surface for any Euler angles, but not
so close that the adsorbate's atoms are placed deep inside the surface.  If
in doubt, generate surfaces with a range of z values and check if they make
physical sense.  The default range is 1.5-2.5 Angstroms.

Note that the molecule's spatial position is determined by particular atom,
which defaults to the first one in the Atoms object (the one on the top of
the xyz file, for example).  You should choose an atom that is near the
center of the molecule.  You can specify a different atom via *mol_index*.

```python
basc = BASC(surface, molecule,
    mol_index = 0,
    zbounds = (1.5, 2.5)
)
```


### Previewing a Particular Configuration

You can export an ASE Atoms object corresponding to an arbitrary point in the
parameter space.  This is useful for making sure that BASC is covering the
right function space.

```
ase.io.write("demo.cif", basc.atoms_from_parameters(x, y, z, p, t, s))
```

where *x*, *y*, *z*, *p*, *t*, and *s* correspond to the spatial and Euler
coordinates of the molecule on the surface.

ASE's _*.cif_ files will open in VESTA, ase-gui, and probably most other
programs capable of viewing _*.cif_ files.  However, I have been told that
Medea does not like ASE's _*.cif_ files, so you may want to export in a
different format if you use Medea.


### Obtaining the Relaxed Surface

There is no one-size-fits-all approach for obtaining a relaxed surface slab.
Most approaches involve constructing and relaxing a slab with some number of
atomic layers and possible constraints.

However, for convenience, we have provided a few utilities under the
`basc.relaxation` namespace that might prove helpful:

 1. *basc.relaxation.fit_lattice_parameters_with_stress_tensor* takes in a
    unit cell and attempts to optimize the lattice parameters of the unit
    cell using stress tensors.  Requires a calculator that supports
    calculating stress tensors (e.g., GPAW using the Plane-Wave mode).

 2. *basc.relaxation.fit_lattice_parameters_with_optimizer* takes in a unit
    cell and minimizes the potential energy as a function of the lattice
    parameters using L-BFGS-B.

 3. *basc.relaxation.relax_surface_cell* cuts an empty surface slab and
    relaxes it.  The keyword arguments *fluid_layers* and *fixed_layers* let
    you specify how many unit cell layers to include in the slab.  Note that
    unit cell layers are not necesarilly the same as "atomic" layers.

 4. *basc.relaxation.perform_relaxation* performs a BFGS structural
    relaxation on whatever ASE Atoms object you pass to it.  It includes
    error recovery for relaxations that fail.

Read the docstrings for more details on the `basc.relaxation` subroutines.


### Tuning the Optimization Procedure

The default settings in BASC should work for most systems.  If you are having
poor convergence results, though, you can tune the optimization parameters.

Bayesian Optimization works great when it has the right settings, and poorly
when it has the wrong ones.  The settings include, among other things, the
*length scales*, *variance* (aka *output scale*), and the number of
iterations.

#### Length Scales

The length scales correspond to how "far" one particular data point
influences other data points.  For example, if you had a molecule at
(0.0, 0.0) on the surface (fractional coordinates), the potential energy at
that point probably has a great influence on the point (0.0, 0.1) but
probably not on (0.0, 0.5).  In this case, your length scale in *y* should
be no smaller than 0.1, but certainly not as large as 0.5.  The same
principle applies to the other dimensions.

BASC uses the following defaults for the length scales:

- *x* and *y*: 1 Angstrom expressed in fractional coordinates depending
    on the size of the surface cell passed to BASC
- *z*: 0.25 Angstroms
- Angles, linear molecules: π/6 Radians
- Angles, non-linear molecules: π/4 Radians

You can set custom lengthscales by setting *basc.lengthscales* equal to a
dictionary such as the following:

```python
# Linear molecules (phi_length=0):
basc.lengthscales = {
    "x": ___,
    "y": ___,
    "z": ___,
    "sph": ___
}

# Non-linear molecules:
basc.lengthscales = {
    "x": ___,
    "y": ___,
    "z": ___,
    "p": ___,  # phi
    "t": ___,  # theta
    "s": ___   # psi
}
```

BASC also provides a mechanism for automatically setting length scales to
data.  BASC will obtain several hundred data points and use BFGS to pick the
length scales that maximize the fit of the GP model to those data:

```python
basc.fit_lengthscales()
```

Even if you plan to use GPAW as the calculator in the main step, you should
use Lennard-Jones here, since Lennard-Jones is much faster and will have
similar length scales to the actual DFT objective function.

After running that function, I highly recommend checking the lengthscales
manually to ensure that they make sense.  We've had mixed results with this
automatic lenghscale optimization, so use it with caution.


#### Variance and Number of Iterations

The main purpose for the variance is to control whether the algorithm
*explores* (tries placing the molecule at a new, untested configuration) or
*exploits* (tries placing the molecule in a configuration near a previous
configuration that was promising).  Generally speaking, larger values of
variance result in more exploration.  In order to make the optimization do
both, BASC by default uses the following function for variance.

> variance = base_variance * exp(1 - (iteration * influence_frac)^2)

If you plot that function versus *iteration*, it should have a high value at
iteration=0, slowly start decreasing, and then approach 0 as *iteration*
approaches infinity.  When *iteration* reaches 1/*influence_frac*, the
function will take on a value of *base_variance*.

*base_variance* should be set such that numbers higher than *base_variance*
tend to correspond to exploration, while numbers smaller than *base_variance*
correspond to exploitation.  If you look at the logs and notice that the
algorithm is evaluating too many points that are close together, you should
increase this value.  In a perfect world, this would be the variance of the
objective function.  However, since the potential energy is not drawn from an
ideal Gaussian distribution, there's no one best way to set this value.  BASC
sets it a default of 10 eV.  One option that is a little bit more principled
would be to set it to a multiple of the variance of the training traces
(demonstrated below).

*influence_frac* is the fraction of the parameter space that one particular
observation influences.  Larger length scales will cause a point to have a
larger area of influence.  As an example, in the *x* direction, the fraction
of influence is calculated by:

> x_frac = length_scale_x / x_max * influence_factor

where *influence_factor* is the number of length scales away that you assume
a point to influence.  It controls the tradeoff between speed (higher values)
and accuracy (lower values).  Higher values result in a faster, more
aggressive search, but one which might get trapped in a local minimum.
Values that are too high may also result in premature termination of the
algorithm.  A reasonable range of values would be [1,3].

The number of iterations defaults to twice of 1/*influence_frac*, such that
the algorithm spend roughly half of the time in exploration mode and half in
exploitation mode.  You can change it directly by setting the *niter* option;
however, setting it directly does *not* change the variance function!  If
using the default variance function, the setting you should modify is
*influence_factor*.

The custom values for these parameters should be passed to `basc.run`:

```python
basc.run(
    influence_factor = 1.5,
    base_variance = (np.std(calculator.traces[:,-1])**2) * 10.
)
```

You are also welcome to pass in a custom function to calculate the variance
and bypass all the "influence fraction" magic explained above.  In this case,
you need to specify a number of iterations directly.  For example, to make
the variance simply the variance of all previous observations of the
objective function, you can do:

```python
basc.run(
    niter = 100,
    variance_function = lambda Y: np.std(Y) ** 2
)
```

