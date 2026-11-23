import xarray as xr
import numpy as np
import LOTUS_regression.predictors.download as download
import LOTUS_regression.predictors as predictors
from LOTUS_regression.regression import mzm_regression


def standard_trends(data, var, anom=True):
    preds = predictors.load_data('pred_baseline_pwlt.csv')
    preds = preds.drop(['linear_pre', 'linear_post', 'qboA', 'qboB'], axis=1)
    preds = preds[np.datetime_as_string(data.time.values[0], unit='M'):np.datetime_as_string(data.time.values[-1], unit='M')].interpolate() 
    preds.index = preds.index.to_period('M')
    preds['linear'] = download.load_linear(inflection=int(np.datetime_as_string(data.time.values[0], unit='Y')))['post']

    results = []
    for alt in data.p.values:
        results_lat = []
        for lat in data.lat.values:
            if data[var].sel(p=alt, lat=lat).dropna('time').count('time').values < 60:
                continue

            binned = data.sel(p=alt, lat=lat).interpolate_na('time').dropna('time')

            p = preds.to_xarray()
            p['time'] = preds.index.to_timestamp()
            matches = np.intersect1d(binned.time.values, p.time.values)
            p = p.sel(time=matches).to_pandas()

            r = mzm_regression(p.to_numpy(), binned[var].values)
            if anom:
                res = xr.Dataset({var: ('time', r['gls_results'].resid)},
                                coords={'p': alt,
                                        'lat': lat,
                                        'time': binned.dropna('time').time.values})
            else:  # add back the constant term
                res = xr.Dataset({var: ('time', r['gls_results'].resid + r['gls_results'].params[3])},
                coords={'p': alt,
                        'lat': lat,
                        'time': binned.dropna('time').time.values})
            results_lat.append(res)
        try:
            results.append(xr.concat(results_lat, dim='lat'))
        except:
            continue
    results = xr.concat(results, dim='p')
    return results


def prep_mls(data_loc, var, anom=True, percent=False):
    data = xr.open_dataset(data_loc).load()
    data.time_center_series.attrs['units'] = 'Days since 01-01-2002'
    data = data.swap_dims({'n_times': 'time_center_series'})
    data = xr.decode_cf(data)
    data = data.rename({'time_center_series': 'time'})
    data = data.swap_dims({'n_lat': 'lat', 'n_z': 'p'})
    data = data.rename({'%s_mean_series' % var : var})
    data = data[[var]]

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
    if anom:
        mean = data.groupby('time.month').mean('time')
        data = data.groupby('time.month') - mean
        if percent: 
            data = data / mean.mean('month')
        detrended = standard_trends(data, var, anom=True)
    else:
        detrended = standard_trends(data, var, anom=False)

    return detrended


def prep_osiris_nox(anom=True, percent=False):
    data = xr.open_dataset('/Volumes/Users/krd020/postdoc/data/sat_data/OSIRIS/osiris_v74_nox_MZM_vmr_plevs_10deg.nc').load()
    data = data.rename({'pressure': 'p', 'latitude': 'lat'})
    data = data[['nox_concentration']]
    data['nox_concentration'] *= 1e9
    data = data.sel(p=slice(150, 3))
    # remove non-qbo variability  
    if anom:
        mean = data.groupby('time.month').mean('time')
        data = data.groupby('time.month') - mean
        if percent: 
            data = data / mean.mean('month')
        detrended = standard_trends(data, 'nox_concentration', anom=True)
    else:
        detrended = standard_trends(data, 'nox_concentration', anom=False)
    return detrended


def prep_osiris_o3(anom=True, percent=False):
    data = xr.open_dataset('/Volumes/Users/krd020/postdoc/data/sat_data/OSIRIS/osiris_v74_ozone_MZM_vmr_plevs_10deg.nc').load()
    data = data.rename({'pressure': 'p', 'latitude': 'lat'})
    data = data[['ozone_vmr']]
    data['ozone_vmr'] *= 1e6
    data = data.sel(p=slice(150, 1))
    # remove non-qbo variability  
    if anom:
        mean = data.groupby('time.month').mean('time')
        data = data.groupby('time.month') - mean
        if percent: 
            data = data / mean.mean('month')
        detrended = standard_trends(data, 'ozone_vmr', anom=True)
    else:
        detrended = standard_trends(data, 'ozone_vmr', anom=False)
    return detrended


def prep_swoosh(anom=True, percent=False):
    data = xr.open_dataset('/Volumes/Users/krd020/postdoc/data/swoosh-v02.72-198401-202602-latpress-2.5deg-L31.nc')
    data = data[['combinedo3q']].load()
    data = data.rename({'level': 'p', 'combinedo3q': 'o3'})
    # data = data.sel(p=slice(300, 0.5))

    # rebin
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
    if anom:
        mean = data.groupby('time.month').mean('time')
        data = data.groupby('time.month') - mean
        if percent: 
            data = data / mean.mean('month')
        detrended = standard_trends(data, 'o3', anom=True)
    else:
        detrended = standard_trends(data, 'o3', anom=False)
    return detrended


def prep_ace(var, anom=True, percent=False):
    data = xr.open_dataset('/Volumes/Users/krd020/postdoc/data/ACE/ace_v5p3_%s_both_MZM_vmr_cmam_plev.nc' % var).load()
    data = data.rename({'latitude': 'lat', 'pressure': 'p'})
    data = data[[var]]
    data = data.sel(time=slice('2004-1', '2026-6'))
    if var in ['O3', 'H2O']:
        data = data * 1e6  # ppm
    else:
        data = data * 1e9  # ppb

    # remove non-qbo variability  
    if anom:
        mean = data.groupby('time.month').mean('time')
        data = data.groupby('time.month') - mean
        if percent: 
            data = data / mean.mean('month')
        detrended = standard_trends(data, var, anom=True)
    else:
        detrended = standard_trends(data, var, anom=False)
    return detrended


def make_ace_noy(anom=True, percent=False):
    monthly_means = []
    for s in ['NO', 'NO2', 'HNO3', 'N2O5', 'ClONO2']:
        data = xr.open_dataset('/Volumes/Users/krd020/postdoc/data/ACE/ace_v5p3_%s_both_MZM_vmr_cmam_plev.nc' % s).load()
        data = data.rename({'latitude': 'lat', 'pressure': 'p'})
        data = data[[s]]
        data = data.sel(time=slice('2004-1', '2026-6'))
        data = data * 1e9  # ppb
        monthly_means.append(data)
    
    monthly_means = xr.merge(monthly_means)

    monthly_means['NOy'] = monthly_means['NO'] + monthly_means['NO2'] + monthly_means['HNO3'] + \
                           (2 * monthly_means['N2O5']) + monthly_means['ClONO2']

    if anom:
        mean = monthly_means.groupby('time.month').mean('time')
        monthly_means = monthly_means.groupby('time.month') - mean
        if percent: 
            monthly_means = monthly_means / mean.mean('month')
        detrended = standard_trends(monthly_means, 'NOy', anom=True)
    else:
        detrended = standard_trends(monthly_means, 'NOy', anom=False)

    return detrended


def make_ace_cly(anom=True, percent=False):
    monthly_means = []
    for s in ['HCl', 'ClONO2']:
        data = xr.open_dataset('/Volumes/Users/krd020/postdoc/data/ACE/ace_v5p3_%s_both_MZM_vmr_cmam_plev.nc' % s).load()
        data = data.rename({'latitude': 'lat', 'pressure': 'p'})
        data = data[[s]]
        data = data.sel(time=slice('2004-1', '2026-6'))
        data = data * 1e9  # ppb
        monthly_means.append(data)
    
    monthly_means = xr.merge(monthly_means)

    monthly_means['Cly'] = monthly_means['HCl'] + monthly_means['ClONO2']

    if anom:
        mean = monthly_means.groupby('time.month').mean('time')
        monthly_means = monthly_means.groupby('time.month') - mean
        if percent: 
            monthly_means = monthly_means / mean.mean('month')
        detrended = standard_trends(monthly_means, 'Cly', anom=True)
    else:
        detrended = standard_trends(monthly_means, 'Cly', anom=False)

    return detrended


def sort_by_phase(data):
    dates = xr.open_dataset('/Users/krd020/Documents/qbo-metrics/reference_dates.nc')
    westerly = dates.westerly_onset_date[10::]
    easterly = dates.easterly_onset_date[10::]

    # bin by qbo phase
    westerly_data = []
    for d in range(len(westerly) - 1):
        westerly_data.append(data.sel(time=slice(westerly[d], easterly[d])))
    westerly_data = xr.concat(westerly_data, dim='time')

    easterly_data = []
    for d in range(len(easterly)):
        easterly_data.append(data.sel(time=slice(easterly[d], westerly[d+1])))
    easterly_data = xr.concat(easterly_data, dim='time')

    sorted_data = xr.concat([westerly_data, easterly_data], dim='phase')
    sorted_data['phase'] = ['westerly', 'easterly']
    return sorted_data


if __name__ == "__main__":

    d = prep_osiris_o3()
    d.to_netcdf('/Volumes/Users/krd020/postdoc/ISSI_QBO/data/OSIRIS_V7.4_O3_MZM_anom_2002-2025_for_qbo_metrics.nc')

    #d = prep_osiris_nox()
    #d.to_netcdf('/Volumes/Users/krd020/postdoc/ISSI_QBO/data/OSIRIS_V7.4_NOx_MZM_anom_2002-2025_for_qbo_metrics.nc')

    """
    for var in ['O3', 'H2O']:
        d = prep_ace(var, anom=False, percent=False)
        d.to_netcdf('/Volumes/Users/krd020/postdoc/ISSI_QBO/data/ACE_V5.3_%s_MZM_2004-2026_for_qbo_metrics.nc' % var)
        s = sort_by_phase(d.sel(lat=0))
        s.to_netcdf('/Volumes/Users/krd020/postdoc/ISSI_QBO/data/ACE_V5.3_%s_MZM_by_phase.nc' % var)

        d = prep_ace(var, anom=True, percent=False)
        d.to_netcdf('/Volumes/Users/krd020/postdoc/ISSI_QBO/data/ACE_V5.3_%s_MZM_anom_2004-2026_for_qbo_metrics.nc' % var)
        s = sort_by_phase(d.sel(lat=0))
        s.to_netcdf('/Volumes/Users/krd020/postdoc/ISSI_QBO/data/ACE_V5.3_%s_MZM_anom_by_phase.nc' % var)

        d = prep_ace(var, anom=True, percent=True)
        d.to_netcdf('/Volumes/Users/krd020/postdoc/ISSI_QBO/data/ACE_V5.3_%s_MZM_anom_percent_2004-2026_for_qbo_metrics.nc' % var)
        s = sort_by_phase(d.sel(lat=0))
        s.to_netcdf('/Volumes/Users/krd020/postdoc/ISSI_QBO/data/ACE_V5.3_%s_MZM_anom_percent_by_phase.nc' % var)


    d = make_ace_cly(anom=False, percent=False)
    d.to_netcdf('/Volumes/Users/krd020/postdoc/ISSI_QBO/data/ACE_V5.3_Cly_MZM_2004-2026_for_qbo_metrics.nc')
    s = sort_by_phase(d.sel(lat=0))
    s.to_netcdf('/Volumes/Users/krd020/postdoc/ISSI_QBO/data/ACE_V5.3_Cly_MZM_by_phase.nc')

    d = make_ace_cly(anom=True, percent=False)
    d.to_netcdf('/Volumes/Users/krd020/postdoc/ISSI_QBO/data/ACE_V5.3_Cly_MZM_anom_2004-2026_for_qbo_metrics.nc')
    s = sort_by_phase(d.sel(lat=0))
    s.to_netcdf('/Volumes/Users/krd020/postdoc/ISSI_QBO/data/ACE_V5.3_Cly_MZM_anom_by_phase.nc')

    d = make_ace_cly(anom=True, percent=True)
    d.to_netcdf('/Volumes/Users/krd020/postdoc/ISSI_QBO/data/ACE_V5.3_Cly_MZM_anom_percent_2004-2026_for_qbo_metrics.nc')
    s = sort_by_phase(d.sel(lat=0))
    s.to_netcdf('/Volumes/Users/krd020/postdoc/ISSI_QBO/data/ACE_V5.3_Cly_MZM_anom_percent_by_phase.nc')


    d = make_ace_noy(anom=False, percent=False)
    d.to_netcdf('/Volumes/Users/krd020/postdoc/ISSI_QBO/data/ACE_V5.3_NOy_MZM_2004-2026_for_qbo_metrics.nc')
    s = sort_by_phase(d.sel(lat=0))
    s.to_netcdf('/Volumes/Users/krd020/postdoc/ISSI_QBO/data/ACE_V5.3_NOy_MZM_by_phase.nc')

    d = make_ace_noy(anom=True, percent=False)
    d.to_netcdf('/Volumes/Users/krd020/postdoc/ISSI_QBO/data/ACE_V5.3_NOy_MZM_anom_2004-2026_for_qbo_metrics.nc')
    s = sort_by_phase(d.sel(lat=0))
    s.to_netcdf('/Volumes/Users/krd020/postdoc/ISSI_QBO/data/ACE_V5.3_NOy_MZM_anom_by_phase.nc')

    d = make_ace_noy(anom=True, percent=True)
    d.to_netcdf('/Volumes/Users/krd020/postdoc/ISSI_QBO/data/ACE_V5.3_NOy_MZM_anom_percent_2004-2026_for_qbo_metrics.nc')
    s = sort_by_phase(d.sel(lat=0))
    s.to_netcdf('/Volumes/Users/krd020/postdoc/ISSI_QBO/data/ACE_V5.3_NOy_MZM_anom_percent_by_phase.nc')




    d = prep_swoosh()
    d.to_netcdf('/Volumes/Users/krd020/postdoc/ISSI_QBO/data/SWOOSH_V2.72_O3_MZM_1984-2026_for_qbo_metrics.nc')


    path = '/Volumes/Users/krd020/postdoc/data/issi_info_qbo_timeseries/'
    for var in ['O3', 'temp', 'HCl', 'H2O', 'CO', 'HNO3', 'N2O', 'HCN']:
       d = prep_mls(path + 'MLS_V6.0_%s_zonal_averages_2004-2026_on_p.nc' % var, var, anom=False, percent=False)
       d.to_netcdf('/Volumes/Users/krd020/postdoc/ISSI_QBO/data/MLS_V6.0_%s_MZM_2004-2026_for_qbo_metrics.nc' % var)
       s = sort_by_phase(d.sel(lat=0))
       s.to_netcdf('/Volumes/Users/krd020/postdoc/ISSI_QBO/data/MLS_V6.0_%s_MZM_by_phase.nc' % var)

       d = prep_mls(path + 'MLS_V6.0_%s_zonal_averages_2004-2026_on_p.nc' % var, var, anom=True, percent=True)
       d.to_netcdf('/Volumes/Users/krd020/postdoc/ISSI_QBO/data/MLS_V6.0_%s_MZM_anom_percent_2004-2026_for_qbo_metrics.nc' % var)
       s = sort_by_phase(d.sel(lat=0))
       s.to_netcdf('/Volumes/Users/krd020/postdoc/ISSI_QBO/data/MLS_V6.0_%s_MZM_anom_percent_by_phase.nc' % var)

       d = prep_mls(path + 'MLS_V6.0_%s_zonal_averages_2004-2026_on_p.nc' % var, var, anom=True, percent=False)
       d.to_netcdf('/Volumes/Users/krd020/postdoc/ISSI_QBO/data/MLS_V6.0_%s_MZM_anom_2004-2026_for_qbo_metrics.nc' % var)
       s = sort_by_phase(d.sel(lat=0))
       s.to_netcdf('/Volumes/Users/krd020/postdoc/ISSI_QBO/data/MLS_V6.0_%s_MZM_anom_by_phase.nc' % var)
    """