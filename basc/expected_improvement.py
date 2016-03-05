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

import GPy
import numpy as np
import scipy.optimize
import scipy.stats

class GPExpectedImprovement(GPy.core.GP):
    def __init__(self, X, Y, kernel, likelihood, mean_function,
                 bounds, seed=0, *args, **kwargs):
        GPy.core.GP.__init__(self, X, Y, kernel, likelihood, mean_function,
                             *args, **kwargs)
        self.bounds = bounds
        self.seed = seed

    def expected_improvement(self, XX):
        minY = np.min(self.Y)
        mu,var = self.predict(XX)
        improvement_pdf = scipy.stats.norm.pdf(minY, mu, np.sqrt(var))
        improvement_cdf = scipy.stats.norm.cdf(minY, mu, np.sqrt(var))
        return np.log((minY-mu)*improvement_cdf + var*improvement_pdf)
    
    def max_expected_improvement(self):
        res = scipy.optimize.differential_evolution(
            lambda x: -self.expected_improvement(x.reshape(1, self.input_dim)),
            bounds = self.bounds,
            seed = self.seed
        )
        if not res.success:
            raise Exception("Could not maximize expected improvement. %s" % res.message)
            return
        return res.x
