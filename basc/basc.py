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

    def __init__(self, relaxed_surf, adsorbate, phi_length=0, mol_index=0,
                 noise_variance=1e-4, seed=0, write_logs=True,
                 xbounds=(0,1), ybounds=(0,1), zbounds=(1.5,2.5)):
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
        """

        self.relaxed_surf = relaxed_surf
        self.adsorbate = adsorbate
        self.mol_index = mol_index
        self.noise_variance = noise_variance
        self.seed = seed
        self.X = np.array([])
        self.Y = np.array([])
        self._training_data = None
        self._calculator = None

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

        # Set default values for the length scales (these will be overwritten
        # by the length scale optimizer).  In a 5-D system (phi_length==0),
        # only x, y, z, and sph are used; in a 6-D system, p, t, and s are
        # user instead of sph.
        self.lengthscales = {
            "x": 0.2, "y": 0.2, "z": 0.5,
            "p": phi_length/4., "t": np.pi/8, "s": np.pi/8,
            "sph": np.pi/8
        }

        # Print header
        if write_logs:
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

    def atoms_from_parameters(self, x, y, z, p, t, s):
        """Return an ASE Atoms object corresponding to the given parameters

        If using the five-dimensional system, pass 0 for {p}.
        """

        adsorbate = self.adsorbate.copy()
        adsorbate.rotate_euler(phi=p, theta=t, psi=s)
        surf = self.relaxed_surf.copy()
        utils.add_adsorbate_fractional(
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

    def objective_fn(self, point, calculator=None):
        """Evaluate the potential energy at {point}.

        -- {calculator} (optional): use this calculator instead of the one
           specified by a call to {set_calculator}.
        """

        # Default to internal calculator
        if calculator is None:
            calculator = self._calculator

        # If calculator is still none, throw an exception
        if calculator is None:
            raise RuntimeError("You must specify a calculator as a keyword "
                               "argument or via the {set_calculator} method")

        surf = self.atoms_from_point(point)
        surf.set_calculator(calculator)
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

    def set_calculator(self, calculator):
        self._calculator = calculator

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
        # FIXME: lengthscale range for z?
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
            kern_x = make_periodic_kernel(
                "p", [3], kernel_variance, self.lengthscales["x"],
                self.bnd_range[3])
            kern_z = make_rbf_kernel(
                "t", [4], kernel_variance, self.lengthscales["z"],
                self.bnd_range[4])
            kern_y = make_periodic_kernel(
                "s", [5], kernel_variance, self.lengthscales["y"],
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

    def auto_gp(self, X, D, ei_probe_n=100, write_logs=True,
                variance_function=None, mean_function=None,
                base_variance=None, lengthscale_influence=None):
        """Make a GP using automatic values for mean and variance

        This function ensures that expected improvement is well-defined
        over the parameter space.

        -- {ei_probe_n} (optional, default 100): the number of SOBOL
           points at which to evaluate the expected improvement to ensure
           that it is well-defined.

        -- {variance_function} (optional): a lambda function taking a list
           of function observations and returning the variance to be used
           by the Gaussian kernels.  You want to use a variance value that
           models the topology of the function. Large values will cause
           more exploration, and small values will cause more exploitation.

        -- {mean_function} (optional): like above, but for the mean rather
           than the variance.

        -- {base_variance} and {lengthscale_influence} (optional): values
           to plug into the default explore-first-exploit-later variance
           scheduling function:
               base_variance * np.exp(1 - (len(D)*influence_frac)**2)
           where {influence_frac} is from {observation_influence_fraction}
           called with the provided {lengthscale_influence}.  If
           {variance_function} is specified, it will take precendence.
        """
        # mu,std = scipy.stats.norm.fit(D)
        # var = std**2
        # nY = (3*len(D))//4
        # mu = np.partition(D, nY)[nY]
        # mu = np.mean(D)
        # var = (mu-np.min(D))**2
        # mu = -186.0
        # var = 100 * np.exp(-(len(D)/70.)**2) + 1.

        if variance_function is not None:
            var = variance_function(D)
        elif base_variance is not None and lengthscale_influence is not None:
            influence_frac = self.observation_influence_fraction(
                lengthscale_influence)
            var = base_variance * np.exp(1 - (len(D)*influence_frac)**2)
            if write_logs: print("influence frac: %f" % influence_frac)
        else:
            var = np.stdev(D)**2

        if mean_function is not None:
            mu = mean_function(D)
        else:
            mu = np.mean(D)

        # Ensure that variance is always positive and nonzero
        if var < np.spacing(1):
            var = np.sqrt(np.spacing(1))

        if write_logs:
            print("mu and var: %.6f, %.6f" % (mu, var))

        eipts = self.sobol_points(ei_probe_n)
        while True:
            gp = self.gp_with(X, D, mu, var)
            if np.median(gp.expected_improvement(eipts)) <= float("-inf"):
                var *= 2
                if write_logs:
                    print("note: doubled variance to %.6f for "
                          "numerical stability" % var)
            else:
                break

        return gp

    def observation_influence_fraction(self, lengthscale_influence=2):
        """What fraction of the parameter space will an observation incluence

        This method is useful for calculating exploration/exploitation
        scheduling tradeoffs.

        -- {lengthscale_influence}: the number of length scales away that an
           observation is assumed to influence.  Smaller values are more
           conservative and will return smaller influence fractions.
        """
        li = lengthscale_influence  # shorthand reference
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

    def fit_lengthscales(self, n=500, calculator=None, write_logs=True,
                         variance_function=None):
        """Use Lennard-Jones and BFGS to optimize the length scales

        The default length scales will not work for every system.  In
        order to fit them to an individual system, we can generate {n}
        data points using a cheap {calculator}, defaulting to Lennard-
        Jones, and then use BFGS to maximize the likelihood of the GP
        while varying the length scales.

        -- {variance_function} (optional, default None): see auto_gp
           for details.
        """

        if calculator is None:
            calculator = LennardJones()

        if write_logs:
            print("BASC: FITTING LENGTH SCALES")
            print("n: %d" % n)
            print("calculator: %s" % str(calculator))

        # Pre-process: generate data (relatively fast if LJ is used)
        X = self.sobol_points(n)
        D = np.array([
            self.objective_fn(point, calculator)
            for point in X
        ]).reshape(n,1)

        mu,std = scipy.stats.norm.fit(D)

        if variance_function is not None:
            var = variance_function(D)
        else:
            var = std**2

        gp = self.gp_with(X, D, mu, var)

        # Run the optimizer (the slower step)
        if write_logs:
            print "\n---\nBefore Optimization:\n"
            print gp

        gp.optimize(messages=write_logs)

        if write_logs:
            print "\n---\nAfter Optimization:\n"
            print gp

        # Save results
        self.lengthscales["x"] = float(gp.kern.x.lengthscales)
        self.lengthscales["y"] = float(gp.kern.y.lengthscales)
        self.lengthscales["z"] = float(gp.kern.z.lengthscale)
        if self.use_sph:
            self.lengthscales["sph"] = float(gp.kern.sph.lengthscale)
        else:
            self.lengthscales["p"] = float(gp.kern.p.lengthscales)
            self.lengthscales["t"] = float(gp.kern.t.lengthscale)
            self.lengthscales["s"] = float(gp.kern.s.lengthscales)

    def run_iteration(self, write_logs=True, nsobol=1, **kwargs):
        """Run an iteration of Bayesian Optimization

        If fewer than {nsobol} iterations had been run previously, default to
        using the next point in the SOBOL sequence instead of maximizing the
        expected improvement.

        Any keyword arguments will be passed to {auto_gp}.
        """

        if len(self.X) < nsobol:
            x = self.sobol_points(nsobol)[len(self.X)]
        else:
            gp = self.auto_gp(self.X, self.Y, write_logs=write_logs, **kwargs)
            x = gp.max_expected_improvement()

        try:
            y = self.objective_fn(x)
        except RuntimeError:
            # "Atoms too close!"
            y = np.max(self.Y)
            if write_logs:
                print("ATOMS TOO CLOSE!  Returning maximum observation")

        self.add_xy(x, y)

        if write_logs:
            print("ITERATION: y=%f, x=%s" % (y, str(x)))

    def best(self):
        """Return the best observation from BASC"""

        iter = np.argmin(self.Y)
        y = self.Y[iter]
        x = self.X[iter]
        atoms = self.atoms_from_point(x)
        return x,y,iter,atoms

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
