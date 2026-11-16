import xarray as xr
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("Agg") 


from qbo_metrics.composite import (compute_composite, extract_events)
from qbo_metrics.metrics import (cycle_coherence, daily_composite_period, descent_rate,
                     latitudinal_width, max_min_amplitude, phase_amplitude,
                     zero_crossing_period_range)
from qbo_metrics.plotting import (plot_composite, plot_cycle_coherence,
                      plot_cycle_summary, plot_descent_rate)
from qbo_metrics.utils import  load_data


# Workflow settings.
EXCLUDED_ONSET_YEARS = (2015, 2020, 2022)
SHEAR_TYPES = ("westerly", "easterly")


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

    for var in ['NO', 'NO2', 'N2O5', 'HNO3', 'HCl', 'CH4', 'N2O', 'CCl2F2', 'ClONO2', 'NOy']:
        data_loc = "/Volumes/Users/krd020/postdoc/ISSI_QBO/data/ACE_V5.3_%s_MZM_2004-2026_for_qbo_metrics.nc" % var

        v = var.lower()

        ds_daily, ds_monthly = load_data(data_loc, variables=[v])

        dates = xr.open_dataset('/Users/krd020/Documents/qbo-metrics/reference_dates.nc')
        westerly_u = xr.open_dataset('/Users/krd020/Documents/qbo-metrics/qbo_composite_westerly_exclude_2015_2020.nc')

        # Build both onset composites with one consistent configuration.
        composites = _get_composites(ds_monthly, dates)
        westerly_composite = composites['westerly'].load()
        easterly_composite = composites['easterly'].load()
        westerly_equatorial = westerly_composite.sel(latitude=0)
        easterly_equatorial = easterly_composite.sel(latitude=0)

        # Plot composite structure.
        fig = plot_composite(
            westerly_equatorial[v], x="lag", x_label="Lag (months)",
            title="Zonal wind (contours) and ACE %s (fill)" % v,
            overlay=westerly_u["u"].sel(latitude=0), overlay_step=10.0,
        )
        plt.savefig('/Volumes/Users/krd020/postdoc/ISSI_QBO/figs/ACE_V5.3_%s_westerly_composite.png' % var)

        fig = plot_composite(
            westerly_composite[v].sel(lag=0),
            x="latitude", x_label="Latitude",
            title="Zonal wind (contours) and ACE %s (fill)" % v,
            overlay=westerly_u["u"].sel(lag=0), overlay_step=10.0,
        )
        plt.savefig('/Volumes/Users/krd020/postdoc/ISSI_QBO/figs/ACE_V5.3_%s_westerly_composite_lag0.png' % var)

        try:
            # Diagnose period, phase and amplitude.
            u_composites = {
                "westerly": westerly_equatorial[v],
                "easterly": easterly_equatorial[v],
            }
            u_eq_monthly = (ds_monthly.sel(pres=slice(0.5, 200), latitude=[0]).mean("latitude").load())
            period_range = zero_crossing_period_range(dates, exclude_years=EXCLUDED_ONSET_YEARS)


            composite_amplitude = max_min_amplitude(westerly_equatorial)
            for name, value in composite_amplitude.data_vars.items():
                print(f"Maximum {name} amplitude: {float(value.max()):.2g}")


            variability_cycles = extract_events(
                u_eq_monthly, dates["westerly_onset_date"],
                detrend=False)[v]
            coherence = cycle_coherence(variability_cycles)
            print(f"Mean cycle/composite coherence: {float(coherence['mean_coherence'].mean()):.2f}")

            fig = plot_cycle_summary(coherence, pres=30.0, variable=v)
            plt.savefig('/Volumes/Users/krd020/postdoc/ISSI_QBO/figs/ACE_V5.3_%s_cycle_summary_30hPa.png' % var)

            fig = plot_cycle_summary(coherence, pres=10.0, variable=v)
            plt.savefig('/Volumes/Users/krd020/postdoc/ISSI_QBO/figs/ACE_V5.3_%s_cycle_summary_10hPa.png' % var)

            dates = dates.rename({'westerly_onset_date': 'westerly', 'easterly_onset_date': 'easterly'})
            # Diagnose cycle variability and zero-wind descent.
            descent_cycle_windows = {
                shear: extract_events(
                    u_eq_monthly, dates[shear],
                    exclude_years=EXCLUDED_ONSET_YEARS,
                    detrend=False)[v]
                for shear in SHEAR_TYPES
            }

            print(f"Mean cycle/composite coherence: {float(coherence['mean_coherence'].mean()):.2f}")
            fig = plot_cycle_coherence(coherence, variable=v)
            plt.savefig('/Volumes/Users/krd020/postdoc/ISSI_QBO/figs/ACE_V5.3_%s_cycle_coherence.png' % var)

            u_composites = {
                "westerly": westerly_equatorial[v],
                "easterly": easterly_equatorial[v],
            }

            descent = descent_rate(u_composites, cycles=descent_cycle_windows)
            fig = plot_descent_rate(descent)
            plt.savefig('/Volumes/Users/krd020/postdoc/ISSI_QBO/figs/ACE_V5.3_%s_descent.png' % var)
        except:
            continue