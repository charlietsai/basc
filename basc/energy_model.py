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
import scipy, scipy.stats

class ConvergenceModel(object):
    """An abstract class for a predictive model based on convergence traces"""
    
    def prior(self):
        raise NotImplementedError
        
    def likelihood(self, i, trace):
        raise NotImplementedError
    
    def apply_likelihood(self, prior, likelihood):
        mu_0 = prior.mean()
        std_0 = prior.std()
        mu_l = likelihood.mean()
        std_l = likelihood.std()
        mu = (mu_0/std_0**2 + mu_l/std_l**2)/(1/std_0**2+1/std_l**2)
        std = np.sqrt(np.reciprocal(1/std_0**2 + 1/std_l**2))
        return scipy.stats.norm(mu, std)
    
    def predict_from_trace(self, trace, **kwargs):
        """The primary public entrypoint for {ConvergenceModel}.
        
        Call this method with a 1-D numpy array.  It will return a normal
        distribution over the final energy value ({scipy.stats.norm}) and
        a list of distributions corresponding to the belief after each
        individual iteration: Pr(E_ifty | O_1...O_i)
        """
        dist = self.prior(**kwargs)
        progress = []
        
        # Update the distribution on each element in the trace
        for i in range(len(trace)):
            
            # Calculate the posterior
            likelihood = self.likelihood(i, trace, **kwargs)
            dist = self.apply_likelihood(dist, likelihood)
            progress.append(dist)
            
        # Return Result
        return dist,progress

    def mean_from_trace(self, trace, **kwargs):
        """Like {predict_from_trace} but returns only the posterior mean."""
        pred,progress = self.predict_from_trace(trace, **kwargs)
        return pred.mean()
    
    @classmethod
    def print_convergence(cls, progress, true):
        """Print the progress of a model over each iteration.

        -- {progress} should be a list of {scipy.stats.norm} distributions,
           such as the one returned by {predict_from_trace}.
        -- {true} should be the true, final converged value, which is used
           in the output for illustrating the convergence progress.
        """
        print(" %2s %6s  %6s  %7s  %9s  %6s" %
              ("It", "PDF", "CDF", "Error", "PostMean", "Stdev"))
        for i,d in enumerate(progress):
            print("%2d: %6.4f  %6.4f  %7.4f  %9.4f  %6.4f" %
                  (i, d.pdf(true), d.cdf(true),
                   (d.mean()-true)/true, d.mean(), d.std()))

class EnergyConvergenceModel(ConvergenceModel):
    """Implementation of {ConvergenceModel} for SCF energy traces"""

    def __init__(self, training_data, noise=0.25, **kwargs):
        """
        -- {training_data} should be a 2-D numpy array where each row is
           an SCF run, and each column is the energy reading at that
           iteration in the SCF run.
        -- {noise} is a constant to be added to the standard deviation of
           predictions made from this model.
        """
        self.data = np.array(training_data)
        self.noise = noise
        return
    
    def prior(self):
        return scipy.stats.norm(*scipy.stats.norm.fit(self.data[:,-1]))
    
    def likelihood(self, i, trace):
        # Model the function as having a certain "bias" or "delta" at
        # each iteration that we learn.
        diff_at_i = self.data[:,-1] - self.data[:,i]
        mu_d,std_d = scipy.stats.norm.fit(diff_at_i)
        mu_d += trace[i]
        
        # Heuristic: add noise to the observation proportional to the
        # difference between this observation and the average of the last
        # three observations.
        noise = self.noise + np.mean(np.abs(trace[max(i-3,0):i+1]-trace[i]))*0.5
        std_d += noise
        
        # Return the likelihood
        return scipy.stats.norm(mu_d, std_d)
