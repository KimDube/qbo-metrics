import xarray as xr
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("Agg") 
plt.style.use('/Users/krd020/Documents/qbo-metrics/plot_style.mplstyle')

target = 30
bad_years = [2015, 2016, 2020, 2021, 2022]

cmap = plt.colormaps['turbo_r']
colors = cmap(np.linspace(0, 1, 9))

for phase in ['westerly', 'easterly']:

    fig, ax = plt.subplots(figsize=(5, 3))

    for idv, var in enumerate(['CH4', 'N2O', 'CCl2F2', 'CCl3F', 'CH3Cl', 'NOy', 'Cly', 'O3', 'H2O']):

        data_anom_percent = xr.open_dataset("/Volumes/Users/krd020/postdoc/ISSI_QBO/data/ACE_V5.3_%s_MZM_anom_percent_by_phase.nc" % var).load()
        data_anom_percent = data_anom_percent.sel(phase=phase)
        data_anom_percent = xr.concat([data_anom_percent.sel(time=slice('2004-1', '2014-12')), data_anom_percent.sel(time=slice('2017-1', '2019-12'))], dim='time')
        data_anom_percent = data_anom_percent.sel(p=target, method='nearest')

        data = xr.open_dataset("/Volumes/Users/krd020/postdoc/ISSI_QBO/data/ACE_V5.3_%s_MZM_by_phase.nc" % var).load()
        data = data.sel(phase=phase)
        data = xr.concat([data.sel(time=slice('2004-1', '2014-12')), data.sel(time=slice('2017-1', '2019-12'))], dim='time')
        data = data.sel(p=slice(90, 1))
        data['logp'] = np.log(data['p']) 
        # flip the pressure levels so that the sign is physically correct when comparing to Chipperfield (e.g. O3 increases with height at these levels)
        grad = [np.gradient(data[var].isel(time=i), np.flipud(data['logp'].values)) / data[var].isel(time=i) for i in range(len(data.time))]
        grad = xr.concat(grad, dim='time')
        grad = grad.sel(p=target, method='nearest')

        # ax.scatter(100*data_anom_percent[var], grad, s=2, label=var, color=colors[idv])
        ax.errorbar(100*data_anom_percent[var].mean('time'), grad.mean('time'), xerr=100*data_anom_percent[var].std('time'), yerr=grad.std('time'),
                     marker='.', label=var, color=colors[idv])

    ax.set_title("QBO-%s, $\sim$%s hPa" % (phase, target))
    ax.set_ylabel("Gradient")
    ax.set_ylim([-3, 3])
    ax.set_xlim([-20, 20])
    ax.grid(True)
    ax.set_xlabel("Anomaly [\%]")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.6))
    plt.subplots_adjust(right=0.78, bottom=0.15)
    plt.savefig('/Volumes/Users/krd020/postdoc/ISSI_QBO/figs/ACE_%s_gradient_vs_anom_%shPa.png' % (phase, target))
    plt.close(fig)


