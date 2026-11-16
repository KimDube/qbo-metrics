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

data_hno3 = xr.open_dataset("/Volumes/Users/krd020/postdoc/ISSI_QBO/data/MLS_V6.0_HNO3_MZM_anom_by_phase.nc")
data_n2o = xr.open_dataset("/Volumes/Users/krd020/postdoc/ISSI_QBO/data/MLS_V6.0_N2O_MZM_anom_by_phase.nc")

fig, ax = plt.subplots(figsize=(7, 3), constrained_layout=True)
ax.plot(data_o3.sel(phase='westerly').time, data_o3.sel(p=30, method='nearest').sel(phase='westerly').O3, color='k', linestyle='', marker='.')
ax.plot(data_o3.sel(phase='easterly').time, data_o3.sel(p=30, method='nearest').sel(phase='easterly').O3, color='blue', linestyle='', marker='.')
for d in dates.westerly_onset_date.values:
    ax.axvline(d, color='red', lw=1, ls="--")
for d in dates.easterly_onset_date.values:
    ax.axvline(d, color='purple', lw=1, ls="--")
ax.set_xlim([pd.to_datetime('2004-1'), pd.to_datetime('2022-12')])
plt.savefig('/Volumes/Users/krd020/postdoc/ISSI_QBO/figs/MLS_V6.0_O3_30hPa.png')

