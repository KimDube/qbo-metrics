import xarray as xr
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("Agg") 
plt.style.use('/Users/krd020/Documents/qbo-metrics/plot_style.mplstyle')


dates = xr.open_dataset('/Users/krd020/Documents/qbo-metrics/reference_dates.nc')
westerly = dates.westerly_onset_date[10::]
easterly = dates.easterly_onset_date[10::]


data_o3 = xr.open_dataset("/Volumes/Users/krd020/postdoc/ISSI_QBO/data/MLS_V6.0_O3_MZM_anom_by_phase.nc")
data_o3 = xr.concat([data_o3.sel(time=slice('2004-1', '2014-12')), data_o3.sel(time=slice('2017-1', '2019-12'))], dim='time')

data_hno3 = xr.open_dataset("/Volumes/Users/krd020/postdoc/ISSI_QBO/data/MLS_V6.0_HNO3_MZM_anom_by_phase.nc")
data_hno3 = xr.concat([data_hno3.sel(time=slice('2004-1', '2014-12')), data_hno3.sel(time=slice('2017-1', '2019-12'))], dim='time')

data_n2o = xr.open_dataset("/Volumes/Users/krd020/postdoc/ISSI_QBO/data/MLS_V6.0_N2O_MZM_anom_by_phase.nc")
data_n2o = xr.concat([data_n2o.sel(time=slice('2004-1', '2014-12')), data_n2o.sel(time=slice('2017-1', '2019-12'))], dim='time')



fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
ax.set_title("30 hPa")
ax.scatter(data_n2o.sel(phase='westerly').sel(p=30, method='nearest').N2O,
            data_o3.sel(phase='westerly').sel(p=30, method='nearest').O3, label='westerly', s=2)
slope, b, r, p, se = stats.linregress(data_n2o.sel(phase='westerly').sel(p=30, method='nearest').N2O.dropna('time'),
                                       data_o3.sel(phase='westerly').sel(p=30, method='nearest').O3.dropna('time'))
ax.plot(np.linspace(-10, 10, 50), np.linspace(-10, 10, 50)*slope + b)

ax.scatter(data_n2o.sel(phase='easterly').sel(p=30, method='nearest').N2O,
            data_o3.sel(phase='easterly').sel(p=30, method='nearest').O3, label='easterly', s=2)
slope, b, r, p, se = stats.linregress(data_n2o.sel(phase='easterly').sel(p=30, method='nearest').N2O.dropna('time'),
                                       data_o3.sel(phase='easterly').sel(p=30, method='nearest').O3.dropna('time'))
ax.plot(np.linspace(-10, 10, 50), np.linspace(-10, 10, 50)*slope + b)

ax.set_ylabel("O3")
ax.set_xlabel("N2O")
plt.legend(frameon=True)
plt.savefig('/Volumes/Users/krd020/postdoc/ISSI_QBO/figs/MLS_V6.0_30hPa_o3_vs_n2o.png')


fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
ax.set_title("30 hPa")
ax.scatter(data_n2o.sel(phase='westerly').sel(p=30, method='nearest').N2O,
            data_hno3.sel(phase='westerly').sel(p=30, method='nearest').HNO3, label='westerly', s=2)
slope, b, r, p, se = stats.linregress(data_n2o.sel(phase='westerly').sel(p=30, method='nearest').N2O.dropna('time'),
                                       data_hno3.sel(phase='westerly').sel(p=30, method='nearest').HNO3.dropna('time'))
ax.plot(np.linspace(-10, 10, 50), np.linspace(-10, 10, 50)*slope + b)
ax.scatter(data_n2o.sel(phase='easterly').sel(p=30, method='nearest').N2O,
            data_hno3.sel(phase='easterly').sel(p=30, method='nearest').HNO3, label='easterly', s=2)
slope, b, r, p, se = stats.linregress(data_n2o.sel(phase='easterly').sel(p=30, method='nearest').N2O.dropna('time'),
                                       data_hno3.sel(phase='easterly').sel(p=30, method='nearest').HNO3.dropna('time'))
ax.plot(np.linspace(-10, 10, 50), np.linspace(-10, 10, 50)*slope + b)
ax.set_ylabel("HNO3")
ax.set_xlabel("N2O")
plt.legend(frameon=True)
plt.savefig('/Volumes/Users/krd020/postdoc/ISSI_QBO/figs/MLS_V6.0_30hPa_hno3_vs_n2o.png')