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
import GPy

class Spherical(GPy.kern.Kern):
    """A kernel operating over a sphere with constant radius.

    This class implements the required methods for a {GPy} kernel."""

    def __init__(self, input_dim, active_dims=None, variance=1., lengthscale=1., name="my_great_circle"):
        super(Spherical, self).__init__(input_dim, active_dims, name)
        self.variance = GPy.core.Param("variance", variance, GPy.core.parameterization.transformations.Logexp())
        self.lengthscale = GPy.core.Param("lengthscale", lengthscale, GPy.core.parameterization.transformations.Logexp())
        self.link_parameters(self.variance, self.lengthscale)
        
    def _unscaled_dist(self, X, X2):
        M = X.shape[0]
        N = X2.shape[0]
        lat1 = (np.pi/2 - X[:,0]).reshape(M,1)
        lat2 = (np.pi/2 - X2[:,0]).reshape(1,N)
        dlat = lat1-lat2
        dlon = X[:,1].reshape(M,1) - X2[:,1].reshape(1,N)
        return 2*np.arcsin(np.sqrt(
            np.square(np.sin(dlat/2)) +
            np.cos(lat1)*np.cos(lat2)*np.square(np.sin(dlon/2))
        ))
    
    def _scaled_dist(self, X, X2):
        return self._unscaled_dist(X, X2) / self.lengthscale
    
    def K(self, X, X2=None):
        if X2 is None: X2 = X
        r = self._scaled_dist(X, X2)
        return self.variance * np.exp(-0.5 * np.square(r))
    
    def Kdiag(self, X):
        return self.variance * np.ones(X.shape[0])
    
    def update_gradients_full(self, dL_dK, X, X2=None):
        if X2 is None: X2 = X
        r = self._scaled_dist(X, X2)
        K = np.exp(-0.5 * np.square(r)) * self.variance
        self.variance.gradient = np.sum(K * dL_dK) / self.variance
        self.lengthscale.gradient = -np.sum(-0.5 * np.square(r) * K * dL_dK) / self.lengthscale
