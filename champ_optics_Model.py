# %%
from glob import glob
from pathlib import PurePath

import numpy as np
import numpy.fft as fft
from scipy import signal
from scipy.optimize import curve_fit

import emcee
from emcee.autocorr import integrated_time

import matplotlib.pyplot as plt
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
    def sin_model(
        z,
        sw_N,
        *params
        ):
        if type(params[0]) == np.ndarray:
            z=np.outer(z, np.ones_like(params[0]))
        sw = []
        for i in range(sw_N):
            amp, l, phi =params[3*i:3*(i+1)]
            sw.append(
                amp*np.sin(2*np.pi*(z/l)+phi)
            )
        sw = np.asarray(sw)
        sw = sw.sum(axis=0)
        return sw
    
    @staticmethod

        
    
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
        self.model_order=Model_order
        self.model_label=model_label
        if Model_order == 2:
            self.model = self.poly_Model_2nd
        elif Model_order == 3:
            self.model = self.poly_Model_3rd
        elif Model_order == 'sin':
            self.model = self.sin_Model

    def model_batch(self, params_sample):
        param_batch_list = [
            params_sample[:,idx] for idx in range(self.param_N)
        ]
        res = self.model(self.x_vec, self.sw_N, *param_batch_list)
        return res

    def set_param_labels(self):
        alpha_beta=['a','b','c','d','e','f','g','h','l','m','n']
        sw_labels = []
        for i in range(self.sw_N):
            sw_labels += [ 
                f'$A_{i}$', fr'$\lambda_{i}$', fr'$\phi_{i}$'
            ]
        self.param_labels = alpha_beta[:self.model_order+1] + sw_labels
        self.param_N = len(self.param_labels)
    
    def log_prob(self, param):
        return self._log_prob(
            param, self.model, self.sw_N, self.param_bounds,
            self.x_vec, self.y_mean, self.y_err
        )
    
    def sample(self, p_0, n_walkers, n_steps):
        bound_width = np.diff(self.param_bounds, axis=-1)[:,0]
        pos = p_0 + np.random.randn(n_walkers, self.param_N) * bound_width*0.01

        self.sampler = emcee.EnsembleSampler(
            n_walkers, self.param_N, 
            self.log_prob,
        )
        self.mcmc_res = self.sampler.run_mcmc(
            pos,
            n_steps,
            progress=True
        )
        return integrated_time(self.sampler.get_chain())

    def get_flatsamp(self, discard, thin):
        self.flat_samples = self.sampler.get_chain(
            discard=discard, thin=thin, flat=True
        )
        print(self.flat_samples.shape)

    def plot_chain(self,):
        fig, axes = plt.subplots(
            self.param_N, figsize=(10, 2.1*self.param_N), 
            sharex=True
        )
        samples = self.sampler.get_chain()
        labels = self.param_labels
        for i in range(self.param_N):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number")
        return fig, axes

    def plot_corner(
            self, 
            discard=100, thin=15,
            **kwds
        ):
        self.get_flatsamp(discard=discard, thin=thin)
        fig = corner.corner(
            self.flat_samples, 
            labels=self.param_labels, 
            **kwds
        )
        return fig

    def plot_residual(self,):
        param_mean = np.atleast_2d(
            self.flat_samples.mean(axis=0)
        )
        pred = self.model_batch(param_mean)[:,0]
        fit_residual_samp = (self.data_samp.T-pred).T
        delta_x = (self.x_vec.max()-self.x_vec.min()) \
            / len(self.x_vec)
        freq = fft.fftshift(
            fft.fftfreq(
                self.x_vec.shape[0], 
                delta_x
            )
        )
        fit_res_spectrum = fft.fftshift(
            fft.fft(fit_residual_samp, axis=0)
        ) / (len(self.x_vec) // 2)
        fig = plt.figure(figsize=(12,10))
        plt.subplot(211)
        mean, err = self.get_mean_err(fit_residual_samp)
        plt.errorbar(
            self.x_vec,
            mean,
            yerr=err,
            marker='.', ls='--', color=f'gray',
            label=self.model_label,
            capsize=3, markeredgewidth=2,
            fillstyle='none'
        )
        plt.hlines(y=0, xmin=0, xmax=12, color='k', ls='--')
        plt.grid(which='both', axis='both')
        plt.legend()
        plt.xlabel(r'$\Delta z$ [mm]')
        plt.ylabel('fitting residual')

        plt.subplot(212)
        mean, err = self.get_mean_err(np.abs(fit_res_spectrum))
        plt.errorbar(
            1./freq,
            mean,
            yerr=err,
            marker='.', ls='--', color=f'gray',
            label=self.model_label,
            capsize=3, markeredgewidth=2,
            fillstyle='none'
        )
        plt.xscale('log')
        plt.yscale('log')
        plt.hlines(y=0, xmin=0, xmax=12, color='k', ls='--')
        plt.grid(which='both', axis='both')
        plt.legend()
        plt.xlabel(r'$\lambda$ [mm]')
        plt.ylabel('residual spectrum')

        plt.tight_layout()
        return fig

    
# %%
