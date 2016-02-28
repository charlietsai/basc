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

from gpaw import GPAW, mpi

from . import BASC
import .utils
import .gpaw_trained

# Constants
lattice = ase.io.read("data/MgS.cif")
molecule_CO = Atoms("CO", positions=[(0.,0.,1.128),(0.,0.,0.)])
job_name = utils.make_name_with_mpi(mpi)
log_dir = "logs/%s" % job_name
gpaw_kwargs = {
	"communicator": mpi.world.new_communicator(list(range(mpi.world.size))),
	"txt": paropen("%s/gpaw.log" % job_name, "a"),
	"spinpol": True
}

# Make the relaxed surface
surf = ase.lattice.general_surface.surface(
    lattice.repeat((2,2,1)), (0,0,1), 2, 15.0)
surf = ase.io.read("data/Fe2O3_1x1_defect_unrelaxed.cif")
c = FixAtoms(indices=list(range(10)))
surf.set_constraint(c)
relaxed_surf = utils.relax_surface_cell(
	surf, GPAW(**kwargs), log_dir, mpi.world.rank==0)

# Make BASC instance and fit length scales
basc = BASC(relaxed_surf, molecule_CO, 0)
basc.fit_lengthscales()

# Obtain training points for GPAWTrained
training_X = basc.sobol_points(5)
training_data = np.array([
	gpaw_trained.obtain_trace(basc.atoms_from_parameters(*point), **kwargs)
	for point in training_X
])
calculator = GPAWTrained(training_data, **gpaw_kwargs)
basc.set_calculator(calculator)

# Run BO iterations
for i in range(200):
	basc.run_iteration()

# All done!
print basc
print basc.X
print basc.Y




