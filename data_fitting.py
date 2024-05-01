# %%
from glob import glob
from pathlib import PurePath
import numpy as np

import numpy.fft as fft
import matplotlib.pyplot as plt

import emcee
from emcee.autocorr import integrated_time
import corner
# %%
def get_mean_err(samples):
    mean = samples.mean(axis=-1)
    std = np.std(samples, axis=-1)
    sample_N = samples.shape[1]
    err = std / np.sqrt(sample_N-1)
    return mean, err
# %%
class MCMC_Model():
    @staticmethod
    def _log_prior(param, bounds):
        logprior = 0
        if np.any(param<bounds[:,0]) or np.any(param >= bounds[:,1]):
            logprior = -np.inf
        return logprior

    @staticmethod
    def _log_prob(param, model, sw_N, bounds, x, y, err):
        logprior = MCMC_Model._log_prior(param, bounds)
        if np.isinf(logprior):
            return logprior
        pred = model(x, sw_N, *param)
        loglike = -0.5* np.sum(((pred-y)/err)**2)
        return loglike+logprior

    @staticmethod
    def model(
        x, 
        sw_N,
        *params
    ):
        if type(params[0]) == np.ndarray:
            x = np.outer(x, np.ones_like(params[0]))
        a,b,c = params[:3]
        sw = []
        for i in range(sw_N):
            amp, l, phi = params[3+3*i:3+3*(i+1)]
            sw.append(
                amp*np.sin(2*np.pi*(x/l)+phi)
            )
        sw = np.asarray(sw)
        sw = sw.sum(axis=0)
        return c + a*(x-b)**2 + sw

    def __init__(
            self, 
            x_vec, data_samp, 
            sw_N,
            param_bounds,
            model_label=""
        ) -> None:
        self.x_vec = x_vec
        self.data_samp = data_samp
        self.y_mean, self.y_err = get_mean_err(self.data_samp)
        self.sw_N = sw_N
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

    def log_prior(self, param):
        return self._log_prior(param, self.param_bounds)

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
        mean, err = get_mean_err(fit_residual_samp)
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
        mean, err = get_mean_err(np.abs(fit_res_spectrum))
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
folder = PurePath('./3_21/')
folder2 = PurePath('./4_24/')
# %%
data_file_list = sorted(glob(str(folder2 / "D_75*")))[:-1]
print(data_file_list)
data_list = [
    np.genfromtxt(data_file) 
        for data_file in data_file_list
]
# %%
fig=plt.figure(figsize=(12,10))
for idx, data in enumerate(data_list):
    x = data[:,0]
    plt.subplot(211)
    plt.plot(
        x,
        data[:,1],
        '*-',
        label=f"{data_file_list[idx].split('/')[-1]}"
    )
    plt.grid(axis='both')
    plt.legend(fontsize=5)
    plt.xlabel(r'$\Delta z$ [mm]')

    plt.subplot(212)
    freq = fft.fftshift(
        fft.fftfreq(
            data.shape[0], 
            np.diff(x).mean()
        )
    )
    delta_x = np.diff(x).mean()
    spectrum = fft.fftshift(fft.fft(data[:,1])) / (len(x)//2)
    plt.plot(
        1./freq,
        np.abs(spectrum),'*-',
        label=f"{data_file_list[idx].split('/')[-1]}"
    )
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(axis='both')
    plt.legend(fontsize=5)
    plt.xlabel(r'$\lambda \ [\mathrm{mm}]$')
# plt.xlim([0,20])
plt.tight_layout()
plt.show()
# %%
data_N = data_list[0].shape[0]
x_vec = data_list[0][:,0]
data_samples = np.zeros([data_N, 3])
for idx, data in enumerate(data_list):
    data_samples[:,idx] = data[:,1]

data_samples_zp = np.zeros([121, 3])
data_samples_zp[-121:,:] = data_samples
freq_vec = fft.fftshift(
    fft.fftfreq(
        data_samples_zp.shape[0], 
        np.diff(x_vec).mean()
    )
)
spec_samp = fft.fftshift(
    fft.fft(data_samples_zp, axis=0)
) / (data_samples_zp.shape[0]//2)
# %%
plt.figure(figsize=[12,10])

plt.subplot(311)
mean, err = get_mean_err(data_samples)
plt.errorbar(
    x_vec,
    mean,
    yerr=err,
    marker='.', ls='--', color=f'gray',
    label=f"all data",
    capsize=3, markeredgewidth=2,
    fillstyle='none'
)
plt.grid(which='both', axis='both')
plt.legend()
plt.xlabel(r'$\Delta z$ [mm]')

plt.subplot(312)
mean, err = get_mean_err(
    np.abs(spec_samp)
)
plt.errorbar(
    1./freq_vec,
    mean,
    yerr=err,
    marker='.', ls='--', color=f'gray',
    label=f"all data",
    capsize=3, markeredgewidth=2,
    fillstyle='none'
)
plt.xscale('log')
plt.yscale('log')

plt.grid(which='both', axis='both')
plt.legend()
plt.xlabel(r'$\lambda \ [\mathrm{mm}]$')

plt.subplot(313)
mean, err = get_mean_err(
    np.abs(spec_samp)
)
plt.errorbar(
    freq_vec,
    mean,
    yerr=err,
    marker='.', ls='--', color=f'gray',
    label=f"all data",
    capsize=3, markeredgewidth=2,
    fillstyle='none'
)
# plt.xscale('log')
plt.yscale('log')

plt.grid(which='both', axis='both')
plt.legend()
plt.xlabel(r'$k \ [\mathrm{mm}^{-1}]$')

plt.tight_layout()
# %%
n_walkers = 50
# %%
bounds_0 = np.asarray(
    [
        [-10, 10],
        [0, 12],
        [1500, 2500]
    ]
)
parabola = MCMC_Model(
    x_vec, data_samples, 
    0,
    bounds_0,
    "parabola"
)
corr_time = parabola.sample(
    np.array([-4.3, 6, 2005]),
    n_walkers,
    5000
)
_ = parabola.plot_chain()
_ = parabola.plot_corner(
    discard=100,
    thin=int(corr_time.max()//2),
    range=[
        [-4.4,-3.8],
        [6.0, 6.2],
        [2000, 2010]
    ]
)
# %%
_ = parabola.plot_residual()
# %%
w_l_1 = 6
bounds_1 = np.asarray(
    [
        [-10, 10],
        [0, 12],
        [1500, 2500],
        [10, 30],
        [5, 7],
        [-np.pi, np.pi],
    ]
)
parabola_sw = MCMC_Model(
    x_vec, data_samples, 
    1,
    bounds_1,
    "parabola + 1 SW"
)
corr_time = parabola_sw.sample(
    np.array([
        -4.3, 6, 2005, 
        15, w_l_1, 0
    ]),
    n_walkers,
    50000
)
# %%
_ = parabola_sw.plot_chain()
# %%
_ = parabola_sw.plot_corner(
    discard=1000,
    thin=int(corr_time.max()//2),
    range=[
        [-5,-4],
        [6.0, 6.5],
        [2000, 2020],
        [10,30],
        [5.1, 6.1],
        [-np.pi, np.pi]
    ]
)
# %%
_ = parabola_sw.plot_residual()
# %%
w_l_1 = 5.6
w_l_2 = 0.95
bounds_2 = np.asarray(
    [
        [-10, 10],
        [0, 12],
        [2000, 2100],
        [0, 25],
        [5, 6],
        [-np.pi, np.pi],
        [10,30],
        [0.9, 1.0],
        [0, 2*np.pi]
    ]
)
parabola_sw_2 = MCMC_Model(
    x_vec, data_samples, 
    2,
    bounds_2,
    "parabola + 2 SW"
)
corr_time = parabola_sw_2.sample(
    np.array([
        -4.3, 6, 2005,
        22, w_l_1, -0.25,
        13, w_l_2, np.pi,
    ]),
    n_walkers,
    50000
)
_ = parabola_sw_2.plot_chain()
# %%
_ = parabola_sw_2.plot_corner(
    discard=1000,
    thin=int(corr_time.max()//2),
    range=[
        [-4.8,-4.25],
        [6.0, 6.4],
        [2000, 2020],
        [0, 25],
        [5, 6],
        [-np.pi, np.pi],
        [10,30],
        [0.9, 1.0],
        [-np.pi, np.pi]
    ]
)
# %%
_ = parabola_sw_2.plot_residual()
# %%
w_l_1 = 5.6
w_l_2 = 0.951
w_l_3 = 3
bounds_3 = np.asarray(
    [
        [-10, 10],
        [0, 12],
        [2000, 2100],
        [0, 25],
        [5, 6],
        [-np.pi, np.pi],
        [10,30],
        [0.9, 1.0],
        [-np.pi, np.pi],
        [0,20],
        [2.5, 4],
        [-np.pi, np.pi]
    ]
)
parabola_sw_3 = MCMC_Model(
    x_vec, data_samples,
    3,
    bounds_3,
    "parabola + 3 SW"
)
corr_time = parabola_sw_3.sample(
    np.array([
        -4.3, 6, 2005,
        22, w_l_1, -0.25,
        14, w_l_2, 2,
        8, w_l_3, -2,
    ]),
    n_walkers,
    100000
)
# %%
_ = parabola_sw_3.plot_chain()
# %%
_ = parabola_sw_3.plot_corner(
    discard=1000,
    thin=int(corr_time.max()//2),
    range=[
        [-4.8,-4.25],
        [6.0, 6.4],
        [2000, 2020],
        [0, 25],
        [5, 6],
        [-np.pi, np.pi],
        [10,30],
        [0.9, 1.0],
        [-np.pi, np.pi],
        [0,20],
        [2.5, 4],
        [-np.pi, np.pi]
    ]
)
# %%
_ = parabola_sw_3.plot_residual()
# %%
w_l_1 = 5.6
w_l_2 = 0.951
w_l_3 = 2
w_l_4 = 3
bounds_4 = np.asarray(
    [
        [-10, 10],
        [0, 12],
        [2000, 2100],
        [0, 25],
        [5, 6],
        [-np.pi, np.pi],
        [10,30],
        [0.9, 1.0],
        [-np.pi, np.pi],
        [0,20],
        [1.5, 2.2],
        [-np.pi, np.pi],
        [0,20],
        [2.5, 4],
        [-np.pi, np.pi]
    ]
)
parabola_sw_4 = MCMC_Model(
    x_vec, data_samples,
    4,
    bounds_4,
    "parabola + 4 SW"
)
corr_time = parabola_sw_4.sample(
    np.array([
        -4.3, 6, 2005,
        22, w_l_1, -0.25,
        14, w_l_2, 2,
        8, w_l_3, -2,
        8, w_l_4, -2,
    ]),
    n_walkers,
    100000
)
# %%
_ = parabola_sw_4.plot_chain()
# %%
_ = parabola_sw_4.plot_corner(
    discard=1000,
    # thin=int(corr_time.max()//2),
    thin=2000,
    range=[
        [-4.8,-4.25],
        [6.0, 6.4],
        [2000, 2020],
        [0, 25],
        [5, 6],
        [-np.pi, np.pi],
        [10,30],
        [0.9, 1.0],
        [-np.pi, np.pi],
        [0,20],
        [1.5, 2.2],
        [-np.pi, np.pi],
        [0,20],
        [2.5, 4],
        [-np.pi, np.pi]
    ]
)
# %%
_ = parabola_sw_4.plot_residual()
# %%
fit_model_list = [
    parabola,
    parabola_sw,
    parabola_sw_2,
    parabola_sw_3,
    parabola_sw_4,
]

b_bins = np.linspace(6.0, 6.4, 1000)
for idx, fit_model in enumerate(fit_model_list):
    _ = plt.hist(
        fit_model.flat_samples[:,1], bins=b_bins,
        density=True,
        alpha=0.3, color=f'C{idx}',
        label=fit_model.model_label
    )
plt.xlabel("b")
plt.ylabel("hist")
plt.legend()

# %%
plt.figure(figsize=(12,10))
plt.subplot(211)
plt.errorbar(
    parabola.x_vec,
    parabola.y_mean,
    yerr=parabola.y_err,
    marker='.', ls='--', color=f'gray',
    label=f"all data",
    capsize=3, markeredgewidth=2,
    fillstyle='none'
)

for idx, fit_model in enumerate(fit_model_list):
    pred_res = fit_model.model_batch(fit_model.flat_samples)
    model_quantil = np.quantile(
        pred_res,
        [0.05, 0.16, 1-0.16, 1-0.05],
        axis=-1
    )
    plt.fill_between(
        fit_model.x_vec,
        y1=model_quantil[0],
        y2=model_quantil[-1],
        color=f"C{idx}",
        alpha=0.2
    )
    plt.fill_between(
        fit_model.x_vec,
        y1=model_quantil[1],
        y2=model_quantil[-2],
        color=f"C{idx}",
        alpha=0.6
    )
plt.grid(which='both', axis='both')
plt.legend()
plt.xlabel(r'$\Delta z$ [mm]')

plt.subplot(212)
for idx, fit_model in enumerate(fit_model_list):
    param_mean = np.atleast_2d(fit_model.flat_samples.mean(axis=0))
    pred_res = fit_model.model_batch(param_mean)[:,0]
    plt.errorbar(
        fit_model.x_vec,
        fit_model.y_mean-pred_res,
        yerr=fit_model.y_err,
        marker='.', ls='--', color=f'C{idx}',
        label=fit_model.model_label,
        capsize=3, markeredgewidth=2,
        fillstyle='none'
    )
plt.hlines(y=0, xmin=0, xmax=12, color='k', ls='--')
plt.grid(which='both', axis='both')
plt.legend()
plt.xlabel(r'$\Delta z$ [mm]')
plt.ylabel('fitting residual')

plt.tight_layout()
# %%
