import xarray as xr
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("Agg") 
plt.style.use('/Users/krd020/Documents/qbo-metrics/plot_style.mplstyle')

from qbo_metrics.utils import load_control_file

SETTINGS = load_control_file()
DATA_LOC = SETTINGS['data_save_path']
FIG_LOC = SETTINGS['fig_save_path']

osiris = xr.open_dataset('/Volumes/Users/krd020/postdoc/data/sat_data/OSIRIS/osiris_v74_ozone_MZM_vmr_plevs_10deg.nc').load()
osiris = osiris.sel(time=slice(SETTINGS['start_yr']+'-'+SETTINGS['start_mo'], SETTINGS['end_yr']+'-'+SETTINGS['end_mo']))

mls = xr.open_dataset('/Volumes/Users/krd020/postdoc/data/issi_info_qbo_timeseries/MLS_V6.0_O3_zonal_averages_2004-2026_on_p.nc')
mls.time_center_series.attrs['units'] = 'Days since 01-01-2002'
mls = mls.swap_dims({'n_times': 'time_center_series'})
mls = xr.decode_cf(mls)
mls = mls.rename({'time_center_series': 'time'})
mls = mls.swap_dims({'n_lat': 'lat', 'n_z': 'p'})
mls = mls.sel(time=slice(SETTINGS['start_yr']+'-'+SETTINGS['start_mo'], SETTINGS['end_yr']+'-'+SETTINGS['end_mo']))

ace = xr.open_dataset('/Volumes/Users/krd020/postdoc/data/ACE/ace_v5p3_O3_both_MZM_vmr_cmam_plev.nc').load()
ace = ace.sel(time=slice(SETTINGS['start_yr']+'-'+SETTINGS['start_mo'], SETTINGS['end_yr']+'-'+SETTINGS['end_mo']))


fig, ax = plt.subplots(figsize=(5, 3))
ax.set_title("2004/8--2025/12", x=0.46, y=1)
ax.plot(mls.lat, 100*mls.n_bin_series.count('time').sel(p=30, method='nearest')/257, label='MLS', marker='.', color='red')
ax.plot(osiris.latitude, 100*osiris.ozone_concentration.count('time').sel(pressure=30)/257, label='OSIRIS', marker='.', color='blue')
ax.plot(ace.latitude, 100*ace.O3.count('time').sel(pressure=30)/257, label='ACE-FTS', marker='.', color='orange')
ax.set_ylabel("Months with data [\\%]")
ax.set_xlabel("Latitude [$^{\\circ}$]")
ax.set_xlim([-80, 80])
ax.grid(True)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(FIG_LOC + 'mls_osiris_ace_months_with_data_30hPa.png')
plt.close(fig)
