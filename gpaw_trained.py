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

from distutils.version import LooseVersion
from gpaw import GPAW, version, wavefunctions
import numpy as np
from .energy_model import EnergyConvergenceModel

class GPAWIterator(GPAW):
    """Extension of GPAW that runs a fixed number of iterations"""
    def __init__(self, niter=1, **kwargs):
        GPAW.__init__(self, **kwargs)
        self.observed_energies = []
        self.niter = niter

    def calculate(self, atoms=None, *args, **kwargs):
        if not atoms:
            atoms = self.atoms

        # Update the atoms and reset if necessary
        if not self.atoms or (self.atoms.positions != atoms.positions).any():
            self.reset_all(atoms)
            GPAW.set_positions(self, atoms)

        # Create the SCF generator if necessary.
        # SCFLoop.run() is an internal method that changed between version
        # 0.11.0 and 0.12.0.  Since older versions of GPAW are still common
        # in the wild, check the version number before making a call.
        if not hasattr(self, "loop") or not self.loop:
            if LooseVersion(version.version_base) < LooseVersion("0.12.0"):
                self.loop = self.scf.run(
                    self.wfs, self.hamiltonian, self.density,
                    self.occupations)
            else:
                self.loop = self.scf.run(
                    self.wfs, self.hamiltonian, self.density,
                    self.occupations, self.forces)

        # Do not perform anything else in this particular method
        return

    def run_iteration(self, *args, **kwargs):
        """Run one SCF iteration and return the iteration number."""
        # Set up the calculator
        self.calculate(*args, **kwargs)

        # Run one iteration.  Force it to perform the calculation by
        # setting "converged" to False.
        self.scf.converged = False
        iter = next(self.loop)
        self.call_observers(iter)
        self.print_iteration(iter)
        self.save(iter)
        return iter

    def run_all_iterations(self, *args, **kwargs):
        """Run all {niter} iterations."""
        while len(self.observed_energies) < self.niter:
            self.run_iteration(*args, **kwargs)

    def set_atoms(self, atoms):
        self.reset_all(atoms)
        GPAW.set_positions(self, atoms)

    def save(self, iter):
        self.observed_energies.append(self.current_energy)

    def reset_all(self, atoms):
        # Based on paw.py calculate() method, case 2
        self.wfs = wavefunctions.base.EmptyWaveFunctions()
        self.occupations = None
        self.density = None
        self.hamiltonian = None
        self.scf = None
        self.loop = None
        self.initialize(atoms)
        self.observed_energies = []

class GPAWTrained(GPAWIterator):
    """A GPAWIterator using a statistical model to reduce SCF iterations
    
    GPAWTrained uses multiple examples of SCF traces from similar
    systems to predict the final converged energy using fewer SCF
    iterations.  For some applications, a single SCF iteration produces
    a sufficiently small confidence interval.

    {get_potential_energy()} is the only public method that is replaced.
    Other methods, like {get_forces()}, are not transformed.
    """

    def __init__(self, training_data, **kwargs):
        GPAWIterator.__init__(self, **kwargs)
        self.energy_model = EnergyConvergenceModel(training_data, **kwargs)

    def get_potential_energy(self, *args, **kwargs):
        """The final energy as predicted by the statistical model."""
        self.run_all_iterations()
        return self.energy_model.mean_from_trace(self.observed_energies)

    @property
    def potential_energy_distribution(self):
        """A normal distribution over the final energy.

        Returns an instance of {scipy.stats.norm}."""
        return self.energy_model.predict_from_trace(self.observed_energies)[0]

    @property
    def current_energy(self):
        """The current energy without any transformations."""
        return GPAW.get_potential_energy(self)


def obtain_trace(atoms, niter=50, **kwargs):
    """Obtain a trace that can be used in {GPAWTrained}"""
    calculator = GPAWIterator(niter, **kwargs)
    atoms.set_calculator(calculator)
    calculator.run_all_iterations()
    return calculator.observed_energies
