import xarray as xr
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("Agg") 


from qbo_metrics.composite import (compute_composite, extract_events)
from qbo_metrics.metrics import (cycle_coherence, descent_rate,
                     max_min_amplitude, phase_amplitude,
                     zero_crossing_period_range)
from qbo_metrics.plotting import (plot_composite, plot_cycle_coherence,
                      plot_cycle_summary, plot_descent_rate,
                      plot_period_fits, plot_phase_amplitude)
from qbo_metrics.utils import load_data, load_control_file


def _get_composites(ds_monthly, dates):
    """Build composites"""
    composites = {}
    for shear in SHEAR_TYPES:
        onset_dates = dates[shear + '_onset_date']
        composites[shear] = compute_composite(
            ds_monthly,
            onset_dates,
            exclude_years=EXCLUDED_ONSET_YEARS,
            detrend=False,
            deseasonalise=False
        )
    return composites


if __name__ == "__main__":

    SETTINGS = load_control_file()
    EXCLUDED_ONSET_YEARS = SETTINGS['EXCLUDED_ONSET_YEARS']
    SHEAR_TYPES = SETTINGS['SHEAR_TYPES']
    DATA_LOC = SETTINGS['data_save_path']
    FIG_LOC = SETTINGS['fig_save_path']
    ROOT_LOC = SETTINGS['root_path']

    for var in SETTINGS['MLS_VARS']:
        if var in ['temp']:
            v = 'T'
        else:
            v = var.lower()

        if var in ['h2o']:  # remove Hunga 
            EXCLUDED_ONSET_YEARS.append(2022)

        ds_daily, ds_monthly = load_data(DATA_LOC + "MLS_V6.0_%s_MZM_anom_2004-2026_for_qbo_metrics.nc" % var, variables=[v])
        ds_monthly = ds_monthly.sel(time=slice(SETTINGS['start_yr']+'-'+SETTINGS['start_mo'], SETTINGS['end_yr']+'-'+SETTINGS['end_mo']))
        ds_monthly = ds_monthly.load()

        dates = xr.open_dataset(ROOT_LOC + 'reference_dates.nc')
        westerly_u = xr.open_dataset(ROOT_LOC + '/qbo_composite_westerly_exclude_2015_2020.nc')
        easterly_u = xr.open_dataset(ROOT_LOC + 'qbo_composite_easterly_exclude_2015_2020.nc')

        # Build both onset composites with one consistent configuration.
        composites = _get_composites(ds_monthly, dates)
        westerly_composite = composites['westerly'].load()
        easterly_composite = composites['easterly'].load()
        westerly_equatorial = westerly_composite.sel(latitude=0)
        easterly_equatorial = easterly_composite.sel(latitude=0)

        westerly_composite.to_netcdf(DATA_LOC + 'MLS_V6.0_%s_westerly_composite.nc' % var)
        easterly_composite.to_netcdf(DATA_LOC + 'MLS_V6.0_%s_easterly_composite.nc' % var)


        # Plot composite structure.
        fig = plot_composite(
            westerly_equatorial[v], x="lag", x_label="Lag (months)",
            title="Zonal wind (contours) and MLS %s (fill)" % v,
            overlay=westerly_u["u"].sel(latitude=0), overlay_step=10.0,
        )
        plt.savefig(FIG_LOC + 'MLS_V6.0_%s_westerly_composite.png' % var)

        fig = plot_composite(
            westerly_composite[v].sel(lag=0),
            x="latitude", x_label="Latitude",
            title="Zonal wind (contours) and MLS %s (fill)" % v,
            overlay=westerly_u["u"].sel(lag=0), overlay_step=10.0,
        )
        plt.savefig(FIG_LOC + 'MLS_V6.0_%s_westerly_composite_lag0.png' % var)

        # Diagnose period, phase and amplitude.
        u_composites = {
            "westerly": westerly_equatorial[v],
            "easterly": easterly_equatorial[v],
        }
        u_eq_monthly = (ds_monthly.sel(pres=slice(0.5, 200), latitude=[0]).mean("latitude").load())
        period_range = zero_crossing_period_range(dates, exclude_years=EXCLUDED_ONSET_YEARS)
        u_cycles = phase_amplitude(u_composites, zero_crossing_range=period_range)
        qbo_period = float(u_cycles["period"])
        # Use the westerly-onset profile in the summary figures.

        cycles = phase_amplitude(
                westerly_equatorial[v], period=qbo_period,
                zero_crossing_range=period_range)
        cycles.to_netcdf(DATA_LOC + 'MLS_V6.0_%s_westerly_eq_cyles.nc' % var)

        cycle_fits = {v: cycles,}
        composite_amplitude = max_min_amplitude(westerly_equatorial)

        fig = plot_period_fits(cycle_fits)
        plt.savefig(FIG_LOC + 'MLS_V6.0_%s_period_fit.png' % var)

        fig = plot_phase_amplitude(cycle_fits)
        plt.savefig(FIG_LOC + 'MLS_V6.0_%s_phase_amplitude.png' % var)

        variability_cycles = extract_events(
            u_eq_monthly, dates["westerly_onset_date"],
            detrend=False)[v]
        coherence = cycle_coherence(variability_cycles)

        coherence.to_netcdf(DATA_LOC + 'MLS_V6.0_%s_cycle_coherence.nc' % var)

        fig = plot_cycle_summary(coherence, pres=30.0, variable=v)
        plt.savefig(FIG_LOC + 'MLS_V6.0_%s_cycle_summary_30hPa.png' % var)

        fig = plot_cycle_summary(coherence, pres=10.0, variable=v)
        plt.savefig(FIG_LOC + '/MLS_V6.0_%s_cycle_summary_10hPa.png' % var)

        dates = dates.rename({'westerly_onset_date': 'westerly', 'easterly_onset_date': 'easterly'})
        # Diagnose cycle variability and zero-wind descent.
        descent_cycle_windows = {
            shear: extract_events(
                u_eq_monthly, dates[shear],
                exclude_years=EXCLUDED_ONSET_YEARS,
                detrend=False)[v]
            for shear in SHEAR_TYPES
        }
        westerly_cycle_fit = u_cycles.sel(shear="westerly")
        westerly_cycle_fit.to_netcdf(DATA_LOC + 'MLS_V6.0_%s_westerly_cycle_fit.nc' % var)

        fig = plot_cycle_coherence(
            coherence, variable=v, vertical_extents=westerly_cycle_fit)
        plt.savefig(FIG_LOC + 'MLS_V6.0_%s_cycle_coherence.png' % var)

        descent = descent_rate(u_composites, cycles=descent_cycle_windows)
        fig = plot_descent_rate(descent)
        plt.savefig(FIG_LOC + 'MLS_V6.0_%s_descent.png' % var)
