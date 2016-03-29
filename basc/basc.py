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

from ase import Atom
from ase.calculators.lj import LennardJones
import GPy
import numpy as np
import scipy.stats

from . import utils
from .expected_improvement import GPExpectedImprovement
from .kernels import Spherical
from .sobol_lib import i4_sobol

class BASC(object):
    """
    This class encapsulates functionality for the majority of operations
    relating to the BASC framework.
    """

    def __init__(self, relaxed_surf, adsorbate, phi_length=2*np.pi,
                 mol_index=0, noise_variance=1e-4, seed=0, verbose=True,
                 xbounds=(0,1), ybounds=(0,1), zbounds=(1.5,2.5),
                 add_adsorbate_method=None):
        """
        -- {relaxed_surf}: An empty surface cell onto which the adsorbate
           molecule can be placed.

        -- {adsorbate}: The desired adsorbate molecule.

        -- {phi_length}: The period over which the "phi" Euler angle is
           periodic. A setting of 0 means that the molecule is symmetric
           about the z axis, which is the case for well-formed CO and CO2 as
           well as single atoms.

        -- {mol_index}: Index of the atom within the adsorbate that should be
           used as the reference point for rotations and placement on the
           surface. An atom near the center of the molecule should be chosen.

        -- {noise_variance}: The variance of the likelihood function in the
           GP. Smaller numbers give better performance, but too small can
           lead to numerical instability.

        -- {seed}: Seed for random number generators and the SOBOL sequence.
           Multiple runs with the same seed will produce the same result.

        -- {xbounds/ybounds}: Bounds for the spatial coordinates.  If using a
           surface in which the unit cell is copied multiple times in x/y,
           you should lower the upper bound for x and y to the point at which
           they are periodic.

        -- {zbounds}: Bounds for the z coordinate.  It should cover a wide
           enough range such that the molecule is sufficiently close to the
           surface for any Euler angles, but not so close that the
           adsorbate's atoms are placed deep inside the surface.

        -- {add_adsorbate_method}: Function to call when adding an adsorbate
           to the surface.  Defaults to basc.utils.add_adsorbate_fractional,
           and any other method should have the same signature.
        """

        self.relaxed_surf = relaxed_surf
        self.adsorbate = adsorbate
        self.mol_index = mol_index
        self.noise_variance = noise_variance
        self.seed = seed
        self.X = np.array([])
        self.Y = np.array([])
        self.verbose = verbose
        self._training_data = None
        self._calculator = None
        self._lengthscales = None

        # Phi length
        if phi_length == 0:
            self.use_sph = True
            self.bounds = np.array([
                xbounds, ybounds, zbounds, # x, y, z
                (0,np.pi), (0,2*np.pi) # incl, azim
            ])
        else:
            self.use_sph = False
            self.bounds = np.array([
                xbounds, ybounds, zbounds, # x, y, z
                (0,phi_length), (0,np.pi), (0,2*np.pi) # p, t, s
            ])

        # Adsorbate method
        if add_adsorbate_method is None:
            add_adsorbate_method = utils.add_adsorbate_fractional
        self.add_adsorbate_method = add_adsorbate_method

        # Print header
        if self.verbose:
            print("BASC: INITIALIZED")
            print("Surface: %s" % self.relaxed_surf.get_name())
            print("Adsorbate: %s" % self.adsorbate.get_name())
            print("phi_length: %f" % phi_length)
            print("mol_index: %d" % mol_index)
            print("noise_variance: %f" % noise_variance)
            print("seed: %d" % seed)
            print("xbounds: %s" % str(xbounds))
            print("ybounds: %s" % str(ybounds))
            print("zbounds: %s" % str(zbounds))
            print("add_adsorbate_method: %s" % str(add_adsorbate_method))

    def atoms_from_parameters(self, x, y, z, p, t, s):
        """Return an ASE Atoms object corresponding to the given parameters

        If using the five-dimensional system, pass 0 for {p}.
        """

        adsorbate = self.adsorbate.copy()
        adsorbate.rotate_euler(phi=p, theta=t, psi=s)
        surf = self.relaxed_surf.copy()
        self.add_adsorbate_method(
            surf, adsorbate, x, y, z, self.mol_index)
        return surf

    def atoms_from_point(self, point):
        """Returns an ASE Atoms object corresponding to the given point

        The difference between {atoms_from_point} and {atoms_from_parameters}
        is that {atoms_from_point} takes the points as a tuple and also
        automatically accounts for the five-dimensional case of CO/CO2.
        """

        if self.use_sph:
            # Note that incl/azim are converted to lat/lon inside of 
            x,y,z,incl,azim = point
            surf = self.atoms_from_parameters(x, y, z, 0, incl, azim)
        else:
            surf = self.atoms_from_parameters(*point)
        return surf

    def objective_fn(self, point):
        """Evaluate the potential energy at {point}.

        -- {calculator} (optional): use this calculator instead of the one
           specified by a call to {set_calculator}.
        """

        surf = self.atoms_from_point(point)
        surf.set_calculator(self.calculator)
        pot = surf.get_potential_energy()
        if pot > 0:
            pot = np.log(pot)
        return pot

    def sobol_points(self, n):
        """Return *n* SOBOL points in the BASC parameter space"""
        if self.use_sph:
            return np.array([
                    i4_sobol(5, i+self.seed)[0] for i in range(n)
                ]) \
                * self.bnd_range + self.bnd_lower
        else:
            return np.array([
                    i4_sobol(6, i+self.seed)[0] for i in range(n)
                ]) \
                * self.bnd_range + self.bnd_lower

    def sobol_atomses(self, n):
        """Return *n* ASE Atoms objects from a SOBOL set"""
        return [
            self.atoms_from_point(point)
            for point in self.sobol_points(n)
        ]

    @property
    def calculator(self):
        """The calculator to use when evaluating the objective function"""

        if self._calculator is None:
            self._calculator = LennardJones()
            if self.verbose:
                print("BASC: Using Lennard-Jones calculator!")

        return self._calculator

    @calculator.setter
    def calculator(self, value):
        self._calculator = value
        if self.verbose:
            print("BASC: Setting calculator to: %s" % str(value))

    @property
    def lengthscales(self):
        """The length scales used by the GP to model the objective function"""

        if self._lengthscales is None:
            self._lengthscales = self.default_lengthscales()
            if self.verbose:
                print("BASC: Using default length scales: %s"
                    % str(self._lengthscales))

        return self._lengthscales

    @lengthscales.setter
    def lengthscales(self, value):
        self._lengthscales = value
        if self.verbose:
            print("BASC: Setting length scales to: %s" % str(value))

    def default_lengthscales(self):
        """Returns an object with default length scales based on cell size"""

        xmin = self.bnd_lower[0]
        xmax = self.bnd_upper[0]
        ymin = self.bnd_lower[1]
        ymax = self.bnd_upper[1]

        # Measure the x-length of the unit cell
        measure_surf = self.relaxed_surf.copy()
        self.add_adsorbate_method(measure_surf, Atom("H"), xmin, ymin, 0, 0)
        self.add_adsorbate_method(measure_surf, Atom("H"), xmax, ymin, 0, 0)
        xdist = measure_surf.get_distance(-1, -2)

        # Measure the y-length of the unit cell
        measure_surf = self.relaxed_surf.copy()
        self.add_adsorbate_method(measure_surf, Atom("H"), xmin, ymin, 0, 0)
        self.add_adsorbate_method(measure_surf, Atom("H"), xmin, ymax, 0, 0)
        ydist = measure_surf.get_distance(-1, -2)

        # Let 1 Angstrom (1E-10 meters) be 1 length scale.  1 Angstrom is
        # a typical bond length in crystal structures.
        xscale = 1. / xdist
        yscale = 1. / ydist

        if self.use_sph:
            return {
                "x": xscale,
                "y": yscale,
                "z": 0.25,
                "sph": np.pi/6
            }

        else:
            return {
                "x": xscale,
                "y": yscale,
                "z": 0.25,
                "p": np.pi/4,
                "t": np.pi/4,
                "s": np.pi/4
            }


    def gp_with(self, X, D, prior_mean, kernel_variance):
        """Return a GP using the BASC kernel and the given data

        If {phi_length} was set to 0 when this class was initialized, then
        the spherical kernel will be used, and the problem will have five
        dimensions; otherwise, standard kernels will be used for the Euler
        coordinates, and there will be six dimensions.

        The prior mean and kernel variance must be specified when calling
        this function; the length scales and likelihood variance (noise)
        are pulled from their corresponding properties on this instance.
        """

        # Construct the kernel
        kern_x = make_periodic_kernel(
            "x", [0], kernel_variance, self.lengthscales["x"],
            self.bnd_range[0])
        kern_y = make_periodic_kernel(
            "y", [1], kernel_variance, self.lengthscales["y"],
            self.bnd_range[1])
        kern_z = make_rbf_kernel(
            "z", [2], kernel_variance, self.lengthscales["z"],
            self.bnd_range[2])

        if self.use_sph:
            kern_sph = make_spherical_kernel(
                "sph", [3,4], kernel_variance, self.lengthscales["sph"])
            kern = kern_x * kern_y * kern_z * kern_sph

        else:
            kern_p = make_periodic_kernel(
                "p", [3], kernel_variance, self.lengthscales["p"],
                self.bnd_range[3])
            kern_t = make_rbf_kernel(
                "t", [4], kernel_variance, self.lengthscales["t"],
                self.bnd_range[4])
            kern_s = make_periodic_kernel(
                "s", [5], kernel_variance, self.lengthscales["s"],
                self.bnd_range[5])
            kern = kern_x * kern_y * kern_z * kern_p * kern_t * kern_s

        # Construct the likelihood and mean function
        likelihood = GPy.likelihoods.Gaussian(variance=self.noise_variance)
        likelihood.variance.constrain_fixed(warning=False)
        mean_function = GPy.mappings.Constant(
            (5 if self.use_sph else 6), 1, prior_mean)
        mean_function.C.constrain_fixed(warning=False)

        # Make GP
        gp = GPExpectedImprovement(
            X, D, kern, likelihood, mean_function, self.bounds,
            seed=self.seed, name="BASC")
        return gp

    def auto_gp(self, X, D, influence_factor=2., base_variance=10.,
                variance_function=None, mean_function=None,
                ei_probe_n=100,):
        """Make a GP using automatic values for mean and variance

        This function ensures that expected improvement is well-defined
        over the parameter space.

        -- {base_variance} (default 10 eV) and {influence_factor}: values
           to plug into the default explore-first-exploit-later variance
           scheduling function:

               base_variance * np.exp(1 - (len(D)*frac)**2)

           where {frac} is from {influence_fraction} called with the
           provided {influence_factor}.  If {variance_function} is specified,
           it will take precendence.

        -- {variance_function} (optional): a lambda function taking a list
           of function observations and returning the variance to be used
           by the Gaussian kernels.  You want to use a variance value that
           models the topology of the function. Large values will cause
           more exploration, and small values will cause more exploitation.

        -- {mean_function} (optional): like above, but for the mean rather
           than the variance.

        -- {ei_probe_n} (optional, default 100): the number of SOBOL
           points at which to evaluate the expected improvement to ensure
           that it is well-defined.
        """

        if variance_function is not None:
            var = variance_function(D)
        else:
            frac = self.influence_fraction(influence_factor)
            var = base_variance * np.exp(1 - (len(D)*frac)**2)

        if mean_function is not None:
            mu = mean_function(D)
        else:
            mu = np.mean(D)

        # Ensure that variance is always positive and nonzero
        if var < np.spacing(1):
            var = np.sqrt(np.spacing(1))

        # Ensure that the variance isn't so small that it leads to numerical
        # instability when calculating the expected improvement
        eipts = self.sobol_points(ei_probe_n)
        while True:
            gp = self.gp_with(X, D, mu, var)
            if np.median(gp.expected_improvement(eipts)) <= float("-inf"):
                var *= 2
                if self.verbose:
                    print("note: doubled variance to %.6f for "
                          "numerical stability" % var)
            else:
                break

        return gp

    def influence_fraction(self, influence_factor):
        """What fraction of the parameter space will an observation incluence

        This method is useful for calculating exploration/exploitation
        scheduling tradeoffs.

        -- {influence_factor}: the number of length scales away that an
           observation is assumed to influence.  Smaller values are more
           conservative and will return smaller influence fractions.
        """
        li = influence_factor  # shorthand reference
        x_frac = self.lengthscales["x"] / self.bnd_range[0] * li
        y_frac = self.lengthscales["y"] / self.bnd_range[1] * li
        z_frac = self.lengthscales["z"] / self.bnd_range[2] * li
        if self.use_sph:
            # Fraction of the area of a sphere cap to the total surface area
            sph_frac = (1 - np.cos(self.lengthscales["sph"]*li)) / 2
            return x_frac * y_frac * z_frac * sph_frac
        else:
            p_frac = self.lengthscales["p"] / self.bnd_range[3] * li
            t_frac = self.lengthscales["t"] / self.bnd_range[4] * li
            s_frac = self.lengthscales["s"] / self.bnd_range[5] * li
            return x_frac * y_frac * z_frac * p_frac * t_frac * s_frac

    def fit_lengthscales(self, n=500, variance_function=None):
        """Use Lennard-Jones and BFGS to optimize the length scales.

        The default length scales will work well for many systems.  However,
        we can fit them to an individual system by maximizing the GP's
        likelihood.  To do this, we can generate {n} data points using a
        cheap calculator, defaulting to Lennard-Jones, and then use BFGS
        to maximize the likelihood of the GP while varying the length scales.

        Caution: Double-check the length scales by hand when using this
        method.  It may give unexpected results.  If in doubt, stick with the
        default length scales, which are based on the size of the unit cell.

        -- {n}: number of points to evaluate.

        -- {variance_function} (optional, default None): see auto_gp
           for details.
        """

        if self.verbose:
            print("BASC: Fitting length scales (n=%d)" % n)

        # Pre-process: generate data (relatively fast if LJ is used)
        X = self.sobol_points(n)
        D = np.array([
            self.objective_fn(point)
            for point in X
        ]).reshape(n,1)

        # Mean/Variance
        mu = np.mean(D)
        if variance_function is not None:
            var = variance_function(D)
        else:
            var = np.std(D)**2

        if self.verbose:
            print("Mean: %f" % mu)
            print("Variance: %f" % var)

        gp = self.gp_with(X, D, mu, var)

        # Run the optimizer (the slower step)
        if self.verbose:
            print("\n---\nBefore Optimization:\n")
            print(gp)

        gp.optimize(messages=self.verbose)

        if self.verbose:
            print("\n---\nAfter Optimization:\n")
            print(gp)

        # Save results
        if self.use_sph:
            self.lengthscales = {
                "x": float(gp.kern.x.lengthscales),
                "y": float(gp.kern.y.lengthscales),
                "z": float(gp.kern.z.lengthscale),
                "sph": float(gp.kern.sph.lengthscale),
            }
        else:
            self.lengthscales = {
                "x": float(gp.kern.x.lengthscales),
                "y": float(gp.kern.y.lengthscales),
                "z": float(gp.kern.z.lengthscale),
                "p": float(gp.kern.p.lengthscales),
                "t": float(gp.kern.t.lengthscale),
                "s": float(gp.kern.s.lengthscales),
            }

    def run_iteration(self, nsobol, i=None, **kwargs):
        """Run a single iteration of Bayesian Optimization

        If fewer than {nsobol} iterations had been run previously, default to
        using the next point in the SOBOL sequence instead of maximizing the
        expected improvement.

        Any keyword arguments will be passed to {auto_gp}.
        """

        if len(self.X) < nsobol:
            x = self.sobol_points(nsobol)[len(self.X)]
        else:
            gp = self.auto_gp(self.X, self.Y, **kwargs)
            x = gp.max_expected_improvement()

        try:
            y = self.objective_fn(x)
        except (RuntimeError, err):
            # Probably "Atoms too close!"
            y = np.max(self.Y)
            if self.verbose:
                print("Runtime exception!  Returning maximum observation")
                print(err)

        self.add_xy(x, y)

        if self.verbose:
            if i is not None:
                print("ITERATION %d: y=%f, x=%s" % (i, y, str(x)))
            else:
                print("ITERATION: y=%f, x=%s" % (y, str(x)))

    def run(self, niter=None, influence_factor=2., nsobol=1, **kwargs):
        """Run BASC to convergence using the default number of iterations.

        Any keyword arguments will be passed to {run_iteration}.
        """

        if niter is None:
            frac = self.influence_fraction(influence_factor)
            niter = int(2/frac)

        if self.verbose:
            print("niter: %d" % niter)
            print("influence_factor: %.2f" % influence_factor)
            print("nsobol: %d" % nsobol)
            print("kwargs: %s" % str(kwargs))

        for i in range(niter):
            self.run_iteration(nsobol, i,
                influence_factor=influence_factor, **kwargs)

        return self.best

    @property
    def best(self):
        """Return the best observation from BASC, a four-tuple:

        -- {point}: the coordinates of the solution in BASC's parameter space
        -- {energy}: the potential energy of the solution
        -- {iter}: the iteration number at which the solution was reached
        -- {atoms}: an ASE Atoms object corresponding to the solution
        """

        iter = np.argmin(self.Y)
        energy = self.Y[iter]
        point = self.X[iter]
        atoms = self.atoms_from_point(point)
        return point,energy,iter,atoms

    def add_xy(self, x, y):
        input_dim = 5 if self.use_sph else 6
        self.X = np.append(self.X, x).reshape(len(self.X)+1, input_dim)
        self.Y = np.vstack(np.append(self.Y, y))

    @property
    def bnd_lower(self):
        """An array with the lower bounds for each parameter"""
        return np.array([b[0] for b in self.bounds])

    @property
    def bnd_upper(self):
        """An array with the upper bounds for each parameter"""
        return np.array([b[1] for b in self.bounds])

    @property
    def bnd_range(self):
        """An array with the range of acceptable values for each parameter"""
        return self.bnd_upper - self.bnd_lower

# Helper functions
def make_periodic_kernel(name, active_dims, variance, lengthscale, range):
    kern = GPy.kern.StdPeriodic(
        input_dim=1, active_dims=active_dims,
        variance=variance, lengthscale=lengthscale,
        wavelength=range,  # i.e., period
        name=name)
    kern.variance.constrain_fixed(warning=False)
    kern.wavelengths.constrain_fixed(warning=False)
    kern.lengthscales.constrain_bounded(range/16., range/1., warning=False)
    return kern
def make_rbf_kernel(name, active_dims, variance, lengthscale, range):
    kern = GPy.kern.RBF(
        input_dim=1, active_dims=active_dims,
        variance=variance, lengthscale=lengthscale,
        name=name)
    kern.variance.constrain_fixed(warning=False)
    kern.lengthscale.constrain_bounded(range/16., range/1., warning=False)
    return kern
def make_spherical_kernel(name, active_dims, variance, lengthscale):
    kern = Spherical(
        input_dim=2, active_dims=active_dims,
        variance=variance, lengthscale=lengthscale,
        name=name)
    kern.variance.constrain_fixed(warning=False)
    kern.lengthscale.constrain_bounded(np.pi/16, np.pi/3, warning=False)
    return kern
