import xarray as xr
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("Agg") 
plt.style.use('/Users/krd020/Documents/qbo-metrics/plot_style.mplstyle')

target = 30

for phase in ['westerly', 'easterly']:

    fig, ax = plt.subplots(figsize=(4, 3))

    for var in ['N2O', 'O3', 'HNO3', 'HCl', 'H2O']:

        data_anom_percent = xr.open_dataset("/Volumes/Users/krd020/postdoc/ISSI_QBO/data/MLS_V6.0_%s_MZM_anom_percent_by_phase.nc" % var).load()
        data_anom_percent = data_anom_percent.sel(time=slice('2004-1', '2014-12'))
        data_anom_percent = data_anom_percent.sel(p=target, method='nearest')
        data_anom_percent = data_anom_percent.sel(phase=phase)

        data = xr.open_dataset("/Volumes/Users/krd020/postdoc/ISSI_QBO/data/MLS_V6.0_%s_MZM_by_phase.nc" % var).load()
        data = data.sel(time=slice('2004-1', '2014-12'))
        data = data.sel(phase=phase)
        data = data.sel(p=slice(90, 1))
        data['logp'] = np.log(data['p']) 
        # flip the pressure levels so that the sign is physically correct when comparing to Chipperfield (e.g. O3 increases with height at these levels)
        grad = [np.gradient(data[var].isel(time=i), np.flipud(data['logp'].values)) / data[var].isel(time=i) for i in range(len(data.time))]
        grad = xr.concat(grad, dim='time')
        grad = grad.sel(p=target, method='nearest')

        ax.scatter(100*data_anom_percent[var], grad, s=2, label=var)


    ax.set_title("QBO-%s, $\sim$%s hPa" % (phase, target))
    ax.set_ylabel("Gradient")
    ax.set_ylim([-4, 4])
    ax.set_xlim([-40, 40])
    ax.grid(True)
    ax.set_xlabel("Anomaly [\%]")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.75))
    plt.subplots_adjust(right=0.78, bottom=0.15)
    plt.savefig('/Volumes/Users/krd020/postdoc/ISSI_QBO/figs/MLS_%s_gradient_vs_anom_%shPa.png' % (phase, target))
    plt.close(fig)


