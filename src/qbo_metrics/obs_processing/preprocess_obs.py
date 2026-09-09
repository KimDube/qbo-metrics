import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("Agg") 
import LOTUS_regression.predictors.download as download
import LOTUS_regression.predictors as predictors
from LOTUS_regression.regression import mzm_regression


def standard_trends(data, var):
    preds = predictors.load_data('pred_baseline_pwlt.csv')
    preds = preds.drop(['linear_pre', 'linear_post', 'qboA', 'qboB'], axis=1)
    preds = preds[np.datetime_as_string(data.time.values[0], unit='M'):np.datetime_as_string(data.time.values[-1], unit='M')].interpolate() 
    preds.index = preds.index.to_period('M')
    preds['linear'] = download.load_linear(inflection=int(np.datetime_as_string(data.time.values[0], unit='Y')))['post']

    results = []
    for alt in data.p.values:
        results_lat = []
        for lat in data.lat.values:
            binned = data.sel(p=alt, lat=lat).interpolate_na('time')

            p = preds.to_xarray()
            p['time'] = preds.index.to_timestamp()
            matches = np.intersect1d(binned.time.values, p.time.values)
            p = p.sel(time=matches).to_pandas()

            r = mzm_regression(p.to_numpy(), binned[var].values)
            res = xr.Dataset({var: ('time', r['gls_results'].resid)},
                            coords={'p': alt,
                                    'lat': lat,
                                    'time': binned.dropna('time').time.values})
            results_lat.append(res)

        results.append(xr.concat(results_lat, dim='lat'))
    results = xr.concat(results, dim='p')
    return results


def prep_mls(data_loc, var, plot=False):
    data = xr.open_dataset(data_loc).load()
    data.time_center_series.attrs['units'] = 'Days since 01-01-2002'
    data = data.swap_dims({'n_times': 'time_center_series'})
    data = xr.decode_cf(data)
    data = data.rename({'time_center_series': 'time'})
    data = data.swap_dims({'n_lat': 'lat', 'n_z': 'p'})
    data = data.rename({'%s_mean_series' % var : var})
    data = data[[var]]

    # rebin
    data = data.sel(p=slice(300, 0.5))
    data = data.sel(time=slice('2004-1', '2026-6'))
    centre_lats = np.arange(-60, 70, 10)
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
    data = data.groupby('time.month') - data.groupby('time.month').mean('time')
    detrended = standard_trends(data, var)

    if plot:
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        im = ax.pcolormesh(data.time, data.p, data[var].sel(lat=0).T, cmap='RdBu_r')
        ax.set_yscale("log")
        ax.invert_yaxis()
        plt.colorbar(im)
        plt.savefig('/Volumes/Users/krd020/postdoc/ISSI_QBO/figs/MLS_v6_%s_ts_trop.png' % var)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        ax.plot(data.time, data[var].sel(p=30, method='nearest').sel(lat=0))
        ax.plot(detrended.time, detrended[var].sel(p=30, method='nearest').sel(lat=0).T)
        plt.savefig('/Volumes/Users/krd020/postdoc/ISSI_QBO/figs/MLS_v6_%s_ts_trop_30hPa.png' % var)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        ax.plot(data.time, data[var].sel(p=50, method='nearest').sel(lat=0))
        ax.plot(detrended.time, detrended[var].sel(p=50, method='nearest').sel(lat=0).T)
        plt.savefig('/Volumes/Users/krd020/postdoc/ISSI_QBO/figs/MLS_v6_%s_ts_trop_50hPa.png' % var)
        plt.close(fig)

    return detrended


def prep_osiris():
    data = xr.open_dataset('/Volumes/Users/krd020/postdoc/data/sat_data/OSIRIS/osiris_v74_nox_MZM_vmr_plevs_10deg.nc').load()
    data = data.rename({'pressure': 'p', 'latitude': 'lat'})
    data = data[['nox_concentration']]
    data['nox_concentration'] *= 1e9
    data = data.sel(time=slice('2002-1', '2021-12'))
    data = data.sel(p=slice(150, 3))
    # remove non-qbo variability  
    data = data.groupby('time.month') - data.groupby('time.month').mean('time')
    detrended = standard_trends(data, 'nox_concentration')
    return detrended


def prep_swoosh():
    data = xr.open_dataset('/Volumes/Users/krd020/postdoc/data/swoosh-v02.72-198401-202602-latpress-2.5deg-L31.nc')
    data = data[['combinedo3q']].load()
    data = data.rename({'level': 'p', 'combinedo3q': 'o3'})
    data = data.sel(p=slice(300, 0.5))

    # rebin
    centre_lats = np.arange(-60, 70, 10)
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
    data = data.groupby('time.month') - data.groupby('time.month').mean('time')
    detrended = standard_trends(data, 'o3')
    return detrended


def prep_ace(var):
    data = xr.open_dataset('/Volumes/Users/krd020/postdoc/data/ACE/ace_v5p3_%s_both_MZM_shift_vmr.nc' % var).load()
    data = data.swap_dims({'latitude': 'lat', 'n_z': 'p'})
    data = data.rename({'%s_mean_series' % var : var})
    data = data[[var]]

    # rebin
    data = data.sel(p=slice(300, 0.5))
    data = data.sel(time=slice('2004-1', '2026-6'))
    centre_lats = np.arange(-60, 70, 10)
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
    data = data.groupby('time.month') - data.groupby('time.month').mean('time')
    detrended = standard_trends(data, var)
    return detrended


if __name__ == "__main__":

    d = prep_ace('CH4')
    d.to_dataset('/Volumes/Users/krd020/postdoc/ISSI_QBO/data/ACE_V5.3_CH4_MZM_2004-2026_for_qbo_metrics.nc')

    d = prep_swoosh()
    d.to_netcdf('/Volumes/Users/krd020/postdoc/ISSI_QBO/data/SWOOSH_V2.72_O3_MZM_1984-2026_for_qbo_metrics.nc')

    # d = prep_osiris()
    # d.to_netcdf('/Volumes/Users/krd020/postdoc/ISSI_QBO/data/OSIRIS_V7.4_NOx_MZM_2002-2021_for_qbo_metrics.nc')

    path = '/Volumes/Users/krd020/postdoc/data/issi_info_qbo_timeseries/'
    for var in ['temp', 'O3', 'HCl', 'H2O', 'CO', 'HNO3', 'N2O', 'HCN']:
       d = prep_mls(path + 'MLS_V6.0_%s_zonal_averages_2004-2026_on_p.nc' % var, var, plot=True)
       d.to_netcdf('/Volumes/Users/krd020/postdoc/ISSI_QBO/data/MLS_V6.0_%s_MZM_2004-2026_for_qbo_metrics.nc' % var)