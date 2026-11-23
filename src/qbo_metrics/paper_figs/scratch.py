import numpy as np
import xarray as xr
from matplotlib import pyplot as plt, ticker
import matplotlib
matplotlib.use("Agg") 
plt.style.use('/Users/krd020/Documents/qbo-metrics/plot_style.mplstyle')
from qbo_metrics.utils import load_control_file



SETTINGS = load_control_file()
DATA_LOC = SETTINGS['data_save_path']
FIG_LOC = SETTINGS['fig_save_path']



mls = xr.open_dataset(DATA_LOC + "MLS_V6.0_O3_MZM_anom_2004-2026_for_qbo_metrics.nc")
osiris = xr.open_dataset(DATA_LOC + "OSIRIS_V7.4_O3_MZM_anom_2002-2025_for_qbo_metrics.nc")
ace = xr.open_dataset(DATA_LOC + "ACE_V5.3_O3_MZM_anom_2004-2026_for_qbo_metrics.nc")

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(mls.time, mls.sel(p=10, method='nearest').sel(lat=0).O3, label='MLS', color='red')
ax.plot(osiris.time, osiris.sel(p=10, method='nearest').sel(lat=0).ozone_concentration, label='OSIRIS', color='blue')
ax.plot(ace.time, ace.sel(p=10, method='nearest').sel(lat=0).O3, label='ACE', color='orange')
ax.set_ylabel("O$_3$ anom. [ppm]")
ax.set_title("10 hPa, 5S-5N")
plt.legend(frameon=True)
plt.savefig('/Volumes/Users/krd020/postdoc/ISSI_QBO/figs/10hpa_lat0_o3_dt_interp.png')


osiris = xr.open_dataset('/Volumes/Users/krd020/postdoc/data/sat_data/OSIRIS/osiris_v74_ozone_MZM_vmr_plevs_10deg.nc').load()
osiris = osiris.rename({'pressure': 'p', 'latitude': 'lat'})
osiris['ozone_vmr'] *= 1e6
mean = osiris.groupby('time.month').mean('time')
osiris = osiris.groupby('time.month') - mean


data = xr.open_dataset('/Volumes/Users/krd020/postdoc/data/issi_info_qbo_timeseries/MLS_V6.0_O3_zonal_averages_2004-2026_on_p.nc').load()
data.time_center_series.attrs['units'] = 'Days since 01-01-2002'
data = data.swap_dims({'n_times': 'time_center_series'})
data = xr.decode_cf(data)
data = data.rename({'time_center_series': 'time'})
data = data.swap_dims({'n_lat': 'lat', 'n_z': 'p'})
data = data.rename({'O3_mean_series' : 'O3'})
data = data[['O3']]

# rebin
data = data.sel(time=slice('2004-1', '2026-6'))
centre_lats = np.arange(-80, 90, 10)
binned_data = []
for lat in centre_lats:
    data_bin = data.sel(lat=slice(lat-5, lat+5))
    weights = np.cos(np.deg2rad(data_bin.lat))
    data_bin = data_bin.weighted(weights).mean(dim='lat')
    data_bin['lat'] = lat
    binned_data.append(data_bin)
data = xr.concat(binned_data, dim='lat')

# remove non-qbo variability  
data = data.resample(time='MS').mean('time')  # switch from 15th to 1st of month to match preds
mean = data.groupby('time.month').mean('time')
data = data.groupby('time.month') - mean


ace = xr.open_dataset('/Volumes/Users/krd020/postdoc/data/ACE/ace_v5p3_%s_both_MZM_vmr_cmam_plev.nc' % 'O3').load()
ace = ace.rename({'latitude': 'lat', 'pressure': 'p'})
ace = ace[['O3']]
ace = ace * 1e6  # ppm
mean = ace.groupby('time.month').mean('time')
ace = ace.groupby('time.month') - mean


fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(data.time, data.sel(p=30, method='nearest').sel(lat=0).O3, label='MLS', color='red')
ax.plot(osiris.time, osiris.sel(p=30, method='nearest').sel(lat=0).ozone_vmr, label='OSIRIS', color='blue')
ax.plot(ace.time, ace.sel(p=30, method='nearest').sel(lat=0).O3, label='ACE', color='orange', marker='.')
ax.set_ylabel("O$_3$ anom [ppm]")
ax.set_title("30 hPa, 5S-5N")
plt.legend(frameon=True)
plt.savefig('/Volumes/Users/krd020/postdoc/ISSI_QBO/figs/30hpa_lat0_o3.png')


fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(data.sel(time='2008-10').sel(lat=0).O3.T, data.p, label='MLS', color='red')
ax.plot(osiris.sel(time='2008-10').sel(lat=0).ozone_vmr.T, osiris.p, label='OSIRIS', color='blue')
#ax.set_ylabel("O$_3$  [ppm]")
#ax.set_title("30 hPa, 5S-5N")
ax.invert_yaxis()
ax.set_yscale("log")
plt.legend(frameon=True)
plt.savefig('/Volumes/Users/krd020/postdoc/ISSI_QBO/figs/profile_lat0_o3.png')
