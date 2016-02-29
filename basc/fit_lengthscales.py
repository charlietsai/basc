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

import numpy as np
import scipy.stats, scipy.optimize
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
from functools import reduce
import itertools
import scipy.spatial
import GPy
import datetime
from .sobol_lib import i4_sobol

import os, sys, csv, errno, pickle, random
from ase import Atoms, Atom
import ase.io
from ase.calculators.lj import LennardJones
sys.path.append("/home/sffc/surfaces")
import automation.compute, automation.downloader, automation.surface
import informatics.util, informatics.name_generator
formula = "Fe2O3"
lattice = ase.io.read("/home/sffc/surfaces/bayes/data/Fe2O3.cif")
relaxed_surf = ase.io.read("/home/sffc/surfaces/bayes/data/Fe2O3_1x1.cif")
molecule_CO = Atoms("CO", positions=[(0.,0.,1.128),(0.,0.,0.)])
from gpaw_freeze_thaw import GPAWTrained
from bayes_optimizer import BayesOptimizer
with np.load("data/fe2o3_relaxed_CO_training.npz") as f:
    training_CO = tuple(f[q] for q in f.files)
fd_grid = np.load("data/fd_grid.npy")



def make_5D_data(n):
    hX = np.array([i4_sobol(5,i)[0] for i in range(n)]) * (1,1,1,np.pi,2*np.pi) + (0,0,1.5,0,0)
    hD = np.array([objective_fn(x,y,z,0,t,s) for x,y,z,t,s in hX]).reshape(n,1)
    mu0,std0 = scipy.stats.norm.fit(hD)
    return hX,hD,mu0,std0

hX,hD,mu0,std0 = make_5D_data(500)
print mu0,std0
print hX.shape,hD.shape

defaults = [.5, .5, 2.0, 0, np.pi]

def get_original():
#     (nIncl,nAzim,lIncl,lAzim,lX) = make_incl_azim(25,25)
#     lY = np.array([objective_fn(defaults[0], defaults[1], defaults[2], 0, t, s) for t,s in lX])
#     gY = lY.reshape(nIncl,nAzim)
    (nXx,nYy,lXx,lYy,lX) = make_xx_yy(21,21)
    lY = np.array([objective_fn(x, y, defaults[2], 0, defaults[3], defaults[4]) for x,y in lX])
    gY = lY.reshape(nXx,nYy).T
    return gY
    
def get_slice(gp):
#     (nIncl,nAzim,lIncl,lAzim,lX) = make_incl_azim(49,49)
#     lX = np.array([(defaults[0], defaults[1], defaults[2], t, s) for t,s in lX])
#     lY,lStd = gp.predict(lX)
#     gY = lY.reshape(nIncl,nAzim)
    (nXx,nYy,lXx,lYy,lX) = make_xx_yy(49,49)
    lX = np.array([(x, y, defaults[2], defaults[3], defaults[4]) for x,y in lX])
    lY,lStd = gp.predict(lX)
    gY = lY.reshape(nXx,nYy).T
    return gY

def plot_slice(gY, *args, **kwargs):
    (nXx,nYy,lXx,lYy,lX) = make_xx_yy(gY.shape[0],gY.shape[1])
    plt.subplot(*args)
    plt.contourf(lXx, lYy, gY, 20, **kwargs)
    plt.grid(True)

original_slice = get_original()
def make_5D_gp(hX, hD, mu0, std0, noise_factor=1e-3, def_lx=0.2, def_ly=0.2, def_lz=0.5, def_ls=np.pi/8):
    # Make Kernel
    variance = (std0**2)
    kernel_x = MyPeriodic(input_dim=1, active_dims=[0], variance=variance, lengthscale=def_lx, period=1, name="kern_x")
    kernel_y = MyPeriodic(input_dim=1, active_dims=[1], variance=variance, lengthscale=def_ly, period=1, name="kern_y")
    kernel_z = GPy.kern.RBF(input_dim=1, active_dims=[2], variance=variance, lengthscale=def_lz, name="kern_z")
    kernel_ts = MyGreatCircle(input_dim=2, active_dims=[3,4], variance=variance, lengthscale=def_ls, name="kern_ts")
    kernel = kernel_x * kernel_y * kernel_z * kernel_ts

    # Make Likelihood and MF
    likelihood = GPy.likelihoods.Gaussian(variance=(std0**2) * noise_factor)
    mean_function = GPy.mappings.Constant(5, 1, mu0)

    # Set Constraints
    kernel_x.variance.constrain_fixed(warning=False)
    kernel_x.period.constrain_fixed(warning=False)
    kernel_x.lengthscale.constrain_bounded(0.01, 1., warning=False)
    kernel_y.variance.constrain_fixed(warning=False)
    kernel_y.period.constrain_fixed(warning=False)
    kernel_y.lengthscale.constrain_bounded(0.01, 1., warning=False)
    kernel_z.variance.constrain_fixed(warning=False)
    kernel_ts.variance.constrain_fixed(warning=False)
    kernel_ts.lengthscale.constrain_bounded(np.pi/16, np.pi/3, warning=False)
    likelihood.variance.constrain_fixed(warning=False)
    mean_function.C.constrain_fixed(warning=False)

    # Make GP
    gp = GPy.core.GP(hX, hD, kernel, likelihood, mean_function)
    return gp

gp = make_5D_gp(hX, hD, mu0, std0)
gp

# Perform Optimization
print gp
pre_optim_slice = get_slice(gp)

gp.optimize(messages=True)

print gp
post_optim_slice = get_slice(gp)





