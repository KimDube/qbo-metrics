import xarray as xr
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("Agg") 
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib import colors, pyplot as plt, ticker

from qbo_metrics.composite import (compute_composite, extract_events)
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
            deseasonalise=False,
            window=1
        )
    return composites


def _plot_reference_pressure(ax, pressure=30.0, label=None, linewidth=1.0):
    """Add the reference pressure to a pressure-axis plot."""
    if pressure is None:
        return None
    return ax.axhline(float(pressure), color="0.3", lw=linewidth, ls="--",
                      alpha=0.8, label=label)





if __name__ == "__main__":

    dates = xr.open_dataset('/Users/krd020/Documents/qbo-metrics/reference_dates.nc')
    westerly_u = xr.open_dataset('/Users/krd020/Documents/qbo-metrics/qbo_composite_westerly_exclude_2015_2020.nc')
    easterly_u = xr.open_dataset('/Users/krd020/Documents/qbo-metrics/qbo_composite_easterly_exclude_2015_2020.nc')

    winds = xr.open_dataset('/Volumes/Users/krd020/postdoc/ISSI_QBO/data/ERA5_daily_zonal_mean.nc')

    latex_names = ['O$_3$', 'T', 'HCl', 'HNO$_3$', 'N$_2$O', 'H$_2$O', 'CO']
    units = ['ppm', 'K', 'ppb', 'ppb', 'ppb', 'ppm', 'ppb']
    levels = [np.arange(-0.6, 0.605, 0.05), np.arange(-5, 5.5, 0.5), np.arange(-0.2, 0.202, 0.02), np.arange(-0.9, 0.91, 0.1), np.arange(-32, 33, 3),
              np.arange(-0.5, 0.505, 0.05), np.arange(-12, 13, 1)]

    months = ['Jan.', 'Feb.', 'Mar.', 'Apr.', 'May', 'Jun.', 'Jul.', 'Aug.', 'Sep.', 'Oct.', 'Nov.', 'Dec.']
    for vid, var in enumerate(['O3', 'temp', 'HCl', 'HNO3', 'N2O', 'H2O', 'CO']):
        data_loc = "/Volumes/Users/krd020/postdoc/ISSI_QBO/data/MLS_V6.0_%s_MZM_anom_2004-2026_for_qbo_metrics.nc" % var

        if var in ['temp']:
            v = 'T'
        else:
            v = var.lower()

        ds_daily, ds_monthly = load_data(data_loc, variables=[v])

        fig = plt.figure(figsize=(7, 6))
        gs = fig.add_gridspec(3, 4, wspace=0.1)
        fig.suptitle("MLS %s" % latex_names[vid], x=0.48, y=0.97)

        for mid, month in enumerate(np.arange(1, 13)):

            ds = ds_monthly.where(ds_monthly.time.dt.month==month, drop=True)

            composites = _get_composites(ds, dates)
            westerly_composite = composites['westerly'].load()
            easterly_composite = composites['easterly'].load()
            westerly_equatorial = westerly_composite.sel(latitude=0)
            easterly_equatorial = easterly_composite.sel(latitude=0)

            ax = plt.subplot(gs[mid])
            da = westerly_composite[v].sel(lag=0)
            lim = float(abs(da).max())

            cs = ax.contourf(da['latitude'], da["pres"], da.transpose("pres", 'latitude'), levels=levels[vid], cmap="RdBu_r", extend='both')
            #overlay=westerly_u["u"].sel(lag=0)
            #overlay_step=10.0
            #olevels = ticker.MultipleLocator(overlay_step).tick_values(float(overlay.min()), float(overlay.max()))
            #widths = np.where(olevels == 0, 1.8, 0.8)
            #cl = ax.contour(overlay["latitude"], overlay["pres"], overlay.transpose("pres", "latitude"),
            #                levels=olevels, colors='k', linewidths=widths)
            #ax.clabel(cl, fontsize=8, fmt="%g")
            ax.set_yscale("log")
            ax.invert_yaxis()
            ax.set_ylim([200, 1])
            ax.set_xlim([-80, 80])
            _plot_reference_pressure(ax, 30)
            if mid > 7:
                ax.set_xlabel("Latitude [$^{\\circ}$]")
            else:
                ax.set_xticklabels([])
            if mid in [0, 4, 8]:
                ax.set_ylabel("Pressure [hPa]")
            else:
                ax.set_yticklabels([])
            ax.set_title(months[mid])

        cb_ax = fig.add_axes([0.85, 0.14, 0.03, 0.76])
        cbar = fig.colorbar(cs, cax=cb_ax)
        cbar.set_label("%s anomaly [%s]" % (latex_names[vid], units[vid]))
        plt.subplots_adjust(top=0.9, right=0.82, left=0.1, bottom=0.14)
        plt.savefig('/Volumes/Users/krd020/postdoc/ISSI_QBO/figs/MLS_V6.0_%s_westerly_composite_lag0_monthly.png' % var)
