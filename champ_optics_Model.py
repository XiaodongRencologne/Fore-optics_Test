# %%
from glob import glob
from pathlib import PurePath

import numpy as np
import numpy.fft as fft
from scipy import signal
from scipy.optimize import curve_fit

import matplotlib.pyplot
import corner

# %%
class MCMC_Model():
    @staticmethod
    def get_mean_err(samples):
        samples=samples/samples.mean(axis=0)
        mean = samples.mean(axis=-1)
        std = np.std(samples, axis=-1)
        sample_N = samples.shape[1]
        err =std/np.sqrt(sample_N-1)
        return mean, err
    @staticmethod
    def poly_Model_2nd(
        z,
        sw_N,
        *params
        ):
        Order=2
        Max=Order+1
        if type(params[0]) == np.ndarray:
            z=np.outer(z, np.ones_like(params[0]))
        a,b,c = params[:Max]
        sw = []
        for i in range(sw_N):
            amp, l, phi = params[Max+Max*i:Max+Max*(i+1)]
            sw.append(
                amp*np.sin(2*np.pi*(z/l)+phi)
            )
        sw = np.asarray(sw)
        sw = sw.sum(axis=0)
        return c + a*(z-b)**2 +sw
    
    @staticmethod
    def poly_Model_3rd(
        z,
        sw_N,
        *params
        ):
        Order=3
        Max=Order+1
        if type(params[0]) == np.ndarray:
            z=np.outer(z, np.ones_like(params[0]))
        a,b,c,d = params[:Max]
        sw = []
        for i in range(sw_N):
            amp, l, phi = params[Max+Max*i:Max+Max*(i+1)]
            sw.append(
                amp*np.sin(2*np.pi*(z/l)+phi)
            )
        sw = np.asarray(sw)
        sw = sw.sum(axis=0)
        return a*z**3 + b*z**2 + c*z + d + sw
    
    @staticmethod
    def _log_prior(param, bounds):
        logprior = 0
        if np.any(param<bounds[:,0]) or np.any(param >= bounds[:,1]):
            logprior = -np.inf
        return logprior
    
    @staticmethod
    def _log_prob(param, model, sw_N, bounds, z, y, err):
        logprior = MCMC_Model._log_prior(param, bounds)
        if np.isinf(logprior):
            return logprior
        pred = model(z, sw_N, *param)
        loglike = -0.5* np.sum(((pred-y)/err)**2)
        return loglike+logprior

    def __init__(
            self,
            x_vec, data_samp,
            Model_order,
            sw_N,
            param_bounds,
            model_label=""
        ) -> None:
        self.x_vec = x_vec
        self.data_samp = data_samp
        self.y_mean, self.y_err = self.get_mean_err(self.data_samp)

        self.sw_N= sw_N
        self.set_param_labels()

        if len(param_bounds) == self.param_N:
            self.param_bounds = param_bounds
        else:
            raise ValueError("param bounds is wrong!")
        self.model_label=model_label

    def model_batch(self, params_sample):
        param_batch_list = [
            params_sample[:,idx] for idx in range(self.param_N)
        ]
        res = self.model(self.x_vec, self.sw_N, *param_batch_list)
        return res

    def set_param_labels(self):
        sw_labels = []
        for i in range(self.sw_N):
            sw_labels += [ 
                f'$A_{i}$', fr'$\lambda_{i}$', fr'$\phi_{i}$'
            ]
        self.param_labels = ['a', 'b', 'c'] + sw_labels
        self.param_N = len(self.param_labels)

    

    