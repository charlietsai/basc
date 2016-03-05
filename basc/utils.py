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
import random

from ase.constraints import FixAtoms
import ase.io
import ase.lattice.general_surface
import ase.lattice.surface
from ase.optimize import BFGS
import numpy as np

COLORS = ["Amaranth","Amber","Amethyst","Apricot","Aquamarine","Azure","Beige","Black","Blue","Blush","Bronze","Brown","Burgundy","Byzantium","Carmine","Cerise","Cerulean","Champagne","Chocolate","Coffee","Copper","Coral","Crimson","Cyan","Emerald","Erin","Gold","Gray","Green","Harlequin","Indigo","Ivory","Jade","Lavender","Lemon","Lilac","Lime","Magenta","Maroon","Mauve","Ocher","Olive","Orange","Orchid","Peach","Pear","Periwinkle","Pink","Plum","Puce","Purple","Raspberry","Red","Rose","Ruby","Salmon","Sangria","Sapphire","Scarlet","Silver","Tan","Taupe","Teal","Turquoise","Violet","Viridian","White"];
ADJECTIVES = ["Adorable","Beautiful","Elegant","Fancy","Glamorous","Handsome","Long","Magnificent","Plain","Quaint","Sparkling","Ugly","Unsightly","Chubby","Crooked","Curved","Skinny","Brave","Calm","Delightful","Eager","Faithful","Gentle","Happy","Jolly","Lively","Obedient","Proud","Relieved","Silly","Thankful","Victorious","Witty","Zealous"];
ANIMALS = ["Aardvarks","Albatrosses","Alligators","Alpacas","Ants","Anteaters","Antelopes","Apes","Armadillos","Baboons","Badgers","Barracudas","Bats","Beavers","Bees","Bison","Butterflies","Camels","Caribous","Cassowaries","Caterpillars","Chamoises","Cheetahs","Chimpanzees","Chinchillas","Cobras","Coyotes","Crabs","Cranes","Crocodiles","Crows","Deer","Dolphins","Doves","Dragonflies","Eagles","Eels","Elephants","Emus","Falcons","Ferrets","Flamingos","Flies","Foxes","Frogs","Gaurs","Gazelles","Gerbils","Giraffes","Gnats","Goldfinches","Goosanders","Geese","Gorillas","Grasshoppers","Guanacos","Gulls","Hamsters","Hares","Hawks","Hedgehogs","Herons","Herrings","Hippopotamuses","Hornets","Hummingbirds","Hyenas","Ibises","Jaguars","Jays","Jellyfish","Kangaroos","Koalas","Larks","Lemurs","Leopards","Lions","Llamas","Lobsters","Lyrebirds","Magpies","Manatees","Mandrills","Minks","Moles","Mongooses","Monkeys","Mooses","Mosquitos","Narwhals","Newts","Nightingales","Octopi","Ostriches","Otters","Owls","Oysters","Panthers","Parrots","Partridges","Pelicans","Penguins","Pheasants","Porcupines","Porpoises","Quail","Raccoons","Ravens","Rhinoceroses","Salamanders","Salmons","Sandpipers","Sardines","Seahorses","Seals","Sharks","Shrews","Skunks","Sloths","Snails","Squirrels","Starlings","Swans","Tigers","Toads","Turtles","Wallabies","Walruses","Wasps","Weasels","Whales","Wolves","Wolverines","Wombats","Wrens","Yaks","Zebras"];

def make_name_seed():
    """Returns a random seed that can be passed to {make_name_with_seed}.

    The seed can be sent to processes running on different cores so that
    all processes can use the same log directories."""
    return [
        random.randrange(len(COLORS)),
        random.randrange(len(ADJECTIVES)),
        random.randrange(len(ANIMALS))
    ]

def make_name_with_seed(seed_arr):
    """Returns a name string based on a seed from {make_name_seed}."""
    color = COLORS[seed_arr[0]]
    adjective = ADJECTIVES[seed_arr[1]]
    animal = ANIMALS[seed_arr[2]]
    return adjective + color + animal;

def make_name():
    """Returns a random name string."""
    return make_with_seed(make_seed())

def make_name_with_mpi(mpi):
    """Returns a random name string that is the same on all processors"""
    name_seed = np.array(make_name_seed())
    mpi.world.broadcast(name_seed, 0)
    return make_name_with_seed(name_seed)

def relax_surface_cell(cell, calculator, log_dir=None, write_logs=True,
                       hkl=(0,0,1), width=1,
                       fluid_layers=1, fixed_layers=1):
    """Run a relaxation procedure on an empty surface."""

    # Print header
    if write_logs:
        print("BASC: RELAXING EMPTY SURFACE CELL")
        print("Compound: %s" % cell.get_name())
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

    # Perform the relaxation
    n = 0
    while True:
        n += 1
        job_prefix = "%s/relax_surface_cell_%000d" % (log_dir, n)
        try: os.mkdir(job_prefix)
        except OSError: pass
        traj_path = "%s/optim.traj" % job_prefix
        bak_path = "%s/backup.pkl" % job_prefix

        if write_logs:
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
            if write_logs:
                print("Encountered LinAlgError; restarting from "
                      "previous BFGS frame")
            surf = ase.io.read(traj_path)

        else:
            # All done
            break

    # Save the final structure
    if write_logs:
        ase.io.write("%s/final.cif" % log_dir, surf)

    return surf

def add_adsorbate_fractional(surf, adsorbate, x, y, z, mol_index):
    """Add an adsorbate to surf at the fractional coordinates x and y."""
    cell_parameters = surf.get_cell()
    ax, ay, az = cell_parameters[0]
    bx, by, bz = cell_parameters[1]
    ase.lattice.surface.add_adsorbate(
        surf, adsorbate, z,
        (
            x*ax + y*bx,
            x*ay + y*by
        ),
        mol_index=mol_index)

