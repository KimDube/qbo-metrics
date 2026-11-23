import xarray as xr
from pathlib import Path
import pandas as pd
import numpy as np
import os
from scipy.interpolate import interp1d
from tqdm import tqdm

k_B = 1.380649e-23  # J/K


cmam_lev = [5.00e+02, 4.50e+02, 4.00e+02,
    3.50e+02, 3.00e+02, 2.85e+02, 2.50e+02, 2.00e+02, 1.70e+02,
    1.50e+02, 1.30e+02, 1.15e+02, 1.00e+02, 9.00e+01, 8.00e+01,
    7.00e+01, 6.00e+01, 5.00e+01, 4.00e+01, 3.50e+01, 3.00e+01,
    2.50e+01, 2.00e+01, 1.50e+01, 1.00e+01, 7.00e+00, 5.00e+00,
    4.00e+00, 3.00e+00, 2.00e+00, 1.50e+00, 1.00e+00, 7.00e-01]

def air_number_density_cm3(P_pa, T_k):
    return P_pa / (k_B * T_k) * 1e-6


def alt_to_pressure_grid(P_alt, q_alt, P_target, log_q=False, axis=-1):
    """
    P_alt    : pressure on the altitude grid (same unit as P_target)
    q_alt    : quantity on the altitude grid (VMR, T, number density, ...)
    P_target : desired pressure grid
    log_q    : interpolate ln(q) instead of q (for quantities spanning orders of magnitude)
    """
    P_alt, q_alt = np.asarray(P_alt, float), np.asarray(q_alt, float)

    # Drop NaNs/invalid values
    ok = np.isfinite(P_alt) & np.isfinite(q_alt) & (P_alt > 0)
    if log_q:
        ok &= q_alt > 0
    lnP, q = np.log(P_alt[ok]), q_alt[ok]
    if log_q:
        q = np.log(q)

    # interp1d with assume_sorted=False sorts x for us (ln P decreases with altitude)
    f = interp1d(lnP, q, kind="linear", bounds_error=False,
                 fill_value=np.nan, assume_sorted=False)
    out = f(np.log(P_target))
    return np.exp(out) if log_q else out


data_path='/Volumes/OsirisDataProducts/OSIRIS/Level2/'
save_path = '/Volumes/Users/krd020/postdoc/data/sat_data/OSIRIS/'

data_loc = data_path + 'o3/7.4/monthly/OZONE-L2-LP-OSIRIS_ODIN-SASK_v7_4-%s.nc'
osiris_v72 = xr.Dataset()
months = pd.date_range('2008-10', '2009-12', freq='MS').strftime('%Y%m')
for m in tqdm(months):
    try:
        df = xr.open_dataset(data_loc % m)
    except:
        print(m)
        continue

    df = df.where((df.local_solar_time < 12) & (df.convergence_ratio < 1.01), drop=True)
    df['ozone_concentration'] = df['ozone_concentration'].where((df.altitude > df.retrieval_lowerbound))
    df['ozone_concentration'] = df['ozone_concentration'].where(df['ozone_concentration_standard_error'] < (df['ozone_concentration']))

    df = df.drop_vars(['_rtm_internal_ozone_concentration', 'vertical_resolution', 'sza', 'ssa', 'saa',
                    'cloud_top_altitude', 'psc_altitude', 'albedo',
                    'retrieval_lowerbound', 'convergence_ratio', 'chi_sq']).load()
    df["n_air"] = air_number_density_cm3(df.pressure * 100, df.temperature)  # if pressure is in hPa

    df['ozone_vmr'] = 6.02214076e+17 * df['ozone_concentration'] / df["n_air"]

    data_on_pres = xr.Dataset()
    for s in df.profile_id.values:
        try:
            vmr_p = alt_to_pressure_grid(df.sel(profile_id=s).pressure,
                                            df.sel(profile_id=s).ozone_vmr, np.array(cmam_lev), log_q=True)
            dp = xr.Dataset({'ozone_vmr': (['pressure'], vmr_p)},
                            coords={'pressure': np.array(cmam_lev),
                                    'profile_id': s,
                                    'latitude': df.sel(profile_id=s).latitude.values,
                                    'longitude': df.sel(profile_id=s).longitude.values,
                                    'time': df.sel(profile_id=s).time.values})
        except:
            continue
        if len(data_on_pres) == 0:
            data_on_pres = dp
        else:
            data_on_pres = xr.concat([data_on_pres, dp], dim='time')

    if len(osiris_v72) == 0:
        osiris_v72 = data_on_pres
    else:
        osiris_v72 = xr.concat([osiris_v72, data_on_pres], dim='time')

vmr_str = "_vmr"
p_str = "_plevs"
osiris_v72 = osiris_v72.sortby('time')
osiris_v72.to_netcdf(save_path+'osiris_v74_ozone%s%s_202611_test.nc' % (vmr_str, p_str))
