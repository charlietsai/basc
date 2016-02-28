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

import .fit_lengthscales
import .utils
from .expected_improvement import GPExpectedImprovement

class BASC(object):
    def __init__(self, relaxed_surf, adsorbate, phi_length=0, seed=0,
                 xbounds=(0,1), ybounds=(0,1), zbounds=(1.5,2.5),
                 noise_variance=1e-4):
        self.relaxed_surf = relaxed_surf
        self.adsorbate = adsorbate
        self.seed = seed
        self.noise_variance = noise_variance
        self.X = np.array()
        self.Y = np.array()
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

    def atoms_from_parameters(self, x, y, z, p, t, s):
        """Return an ASE Atoms object corresponding to the given parameters

        If using the five-dimensional system, pass 0 for {p}."""

        adsorbate = self.adsorbate.copy()
        adsorbate.rotate_euler(phi=p, theta=t, psi=s)
        surf = self.relaxed_surf.copy()
        utils.add_adsorbate_fractional(
            surf, self.adsorbate, x, y, z, self.mol_index)
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

        surf = self.atoms_from_parameters(*point)
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
        kern_x = utils.make_make_periodic_kernel(
            "x", [0], kernel_variance, self.lengthscales["x"],
            self.bnd_range[0])
        kern_y = utils.make_make_periodic_kernel(
            "y", [1], kernel_variance, self.lengthscales["y"],
            self.bnd_range[1])
        kern_z = utils.make_rbf_kernel(
            "z", [2], kernel_variance, self.lengthscales["z"],
            self.bnd_range[2])

        if self.use_sph:
            kern_sph = utils.make_spherical_kernel(
                "sph", [3,4], kernel_variance, self.lengthscales["sph"])
            kern = kern_x * kern_y * kern_z * kern_sph

        else:
            kern_x = utils.make_make_periodic_kernel(
                "p", [3], kernel_variance, self.lengthscales["x"],
                self.bnd_range[3])
            kern_z = utils.make_rbf_kernel(
                "t", [4], kernel_variance, self.lengthscales["z"],
                self.bnd_range[4])
            kern_y = utils.make_make_periodic_kernel(
                "s", [5], kernel_variance, self.lengthscales["y"],
                self.bnd_range[5])
            kern = kern_x * kern_y * kern_z * kern_p * kern_t * kern_s

        # Construct the likelihood and mean function
        likelihood = GPy.likelihoods.Gaussian(variance=noise_variance)
        likelihood.variance.constrain_fixed(warning=False)
        mean_function = GPy.mappings.Constant(
            (5 if self.use_sph else 6), 1, prior_mean)
        mean_function.C.constrain_fixed(warning=False)

        # Make GP
        gp = GPExpectedImprovement(
            X, D, kern, likelihood, mean_function, self.bounds,
            name="BASC")
        return gp

    def auto_gp(self, X, D, variance_transform=None, ei_probe_n=100):
        """Make a GP using automatic values for mean and variance

        This function ensures that expected improvement is well-defined
        over the parameter space.

        -- {variance_transform} (optional, default None): a lambda
           function taking a variance and returning the variance
           transformed by an arbitrary function.  For example, if one
           wanted to exaggerate small variance values, one could pass:
               lambda v: v*(100+v)/(1+v)

        -- {ei_probe_n} (optional, default 100): the number of SOBOL
           points at which to evaluate the expected improvement to ensure
           that it is well-defined.
        """
        mu,std = scipy.stats.norm.fit(D)
        var = std**2
        if variance_transform is not None:
            var = variance_transform(var)

        print("MGP: Mu and Var: %.6f, %.6f" % (mu, var))

        eipts = self.sobol_points(ei_probe_n)
        while True:
            gp = gp_with(X, D, mu, var)
            if np.median(gp.expected_improvement(eipts)) <= float("-inf"):
                print("MGP: Doubling variance for EI")
                var *= 2
            else:
                break

        return gp

    def fit_lengthscales(self, n=500, calculator=None):
        """Use Lennard-Jones and BFGS to optimize the length scales

        The default length scales will not work for every system.  In
        order to fit them to an individual system, we can generate {n}
        data points using a cheap {calculator}, defaulting to Lennard-
        Jones, and then use BFGS to maximize the likelihood of the GP
        while varying the length scales.
        """

        if calculator is None:
            calculator = LennardJones()

        # Pre-process: generate data (relatively fast if LJ is used)
        X = self.sobol_points(n)
        if self.use_sph:
            D = np.array([
                    self.objective_fn((x,y,z,0,t,s), calculator)
                    for x,y,z,t,s in X
                ]).reshape(n,1)
        else:
            D = np.array([
                    self.objective_fn((x,y,z,p,t,s), calculator)
                    for x,y,z,p,t,s in X
                ]).reshape(n,1)

        mu,std = scipy.stats.norm.fit(D)
        gp = self.gp_with(X, D, mu, std0**2, std0**2 * self.noise_factor)

        # Run the optimizer (the slow step)
        print gp
        gp.optimize()
        print gp

        # Save results
        self.lengthscales["x"] = gp.x.lengthscale
        self.lengthscales["y"] = gp.y.lengthscale
        self.lengthscales["z"] = gp.z.lengthscale
        if self.use_sph:
            self.lengthscales["sph"] = gp.sph.lengthscale
        else:
            self.lengthscales["p"] = gp.p.lengthscale
            self.lengthscales["t"] = gp.t.lengthscale
            self.lengthscales["s"] = gp.s.lengthscale

    def run_iteration(self, **kwargs):
        """Run an iteration of Bayesian Optimization

        If no iterations had been run previously, default to using the
        first point from the SOBOL sequence.

        Any keyword arguments will be passed to {auto_gp}.
        """

        if len(self.X) == 0:
            x = self.sobol_points(1)[0]
        else:
            gp = self.auto_gp(self.X, self.Y, **kwargs)
            x = gp.max_expected_improvement()
        y = self.objective_fn(x)
        self.add_xy(x, y)

    def add_xy(self, x, y):
        self.X = np.append(self.X, x).reshape(len(self.X)+1, self.input_dim)
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


