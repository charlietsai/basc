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
3. [GPy](https://github.com/SheffieldML/GPy),
    available in the pip package "gpy"
4. A Python runtime that supportes these packages (e.g. Python 2.7)

I recommend running BASC on Linux, although it should run on any platform that
supports these dependencies.

## Overview

BASC is split into four steps:

1. Generate an empty, relaxed surface
2. Fit the length scales for the GP
3. Obtain training data for accelerated DFT calculations (optional)
4. Run the bayesian optimization

In general, step 1 is the slowest, because it requires performing a full
relaxation with DFT and BFGS (appx 40 calculations).  Step 2 is quick since
we can use a Lennard-Jones potential.  Step 3 is required to use the
"GPAWTrained" calculator for accelerated DFT calculations; it requires
several (at least 5) full DFT calculations.  Step 4 is the actual
optimization step; it is reasonably quick if using GPAWTrained, and somewhat
slower if using a normal DFT calculator.

Included in the top-level directory are four Python script files,
corresponding to the four steps.  There is also a bash script file that
automatically runs all four Python scripts using MPI.

You should use the procedure with four standalone steps if you intend to run
BASC multiple times on the same surface (e.g., to tune the BASC parameters
or run BASC with multiple different adsorbate molecules).  However, you can
use the Bash script if you want a quick start black box.

General settings for the pre-built demo scripts can be customized in
*BASC.ini*.

## Tips for Bayesian Optimization

Bayesian Optimization works great when it has the right settings, and poorly
when it has the wrong ones.  The settings include, among other things, the
*length scales* and the *variance*.

In BASC, we have provided an automated approach to setting both values, which
we hope will work in most scenarios.  However, it is helpful to verify that
the length scales and variance make sense.  You can find them in the BASC log
files for your system.

### Length Scales

Step 2 involves fitting the length scales.  It is helpful to check the
results by hand before continuing to later steps.  The length scales
correspond to how "far" one particular data point influences other data
points.

For example, if you had a molecule at (0.0, 0.0) on the surface, the
potential energy at that point probably has a great influence on the point
(0.0, 0.1) but probably not on (0.0, 0.5).  In this case, your length scale
in *y* should be no smaller than 0.1, but certainly not as large as 0.5.  The
same principle applies to the other dimensions.

### Variance

The main purpose for the variance is to control whether the algorithm
*explores* (tries placing the molecule at a new, untested configuration) or
*exploits* (tries placing the molecule in a configuration near a previous
configuration that was promising).  Generally speaking, larger values of
variance result in more exploration.  In order to make the optimization do
both, BASC by default uses the following function for variance.

    variance = base_variance * exp(1 - (iteration * influence_frac)^2)

If you plot that function versus "iteration", it should have a high value at
iteration=0, slowly start decreasing, and then approach 0 as iteration
approaches infinity.  When the iteration number reaches 1/influence_frac, the
function should take on a value of base_variance.

The "base_variance" is a value learned from the training data in step 3
(square of its standard deviation).  We also multiply base_variance by a
value called "variance_ratio" to give you more control.

The "influence_frac" is the fraction of the parameter space that one
particular observation influences.  (Larger length scales will cause a point
to have a larger area of influence.)  As an example, in the *x* direction,
the fraction of influence is calculated by:

		x_frac = length_scale_x / x_max * lengthscale_influence

where "lengthscale_influence" is the number of length scales away that you
assume a point to influence.  I set lengthscale_influence as 2.5.
