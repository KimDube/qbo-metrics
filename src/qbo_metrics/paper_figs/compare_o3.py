import numpy as np
import xarray as xr
from matplotlib import colors, pyplot as plt, ticker
import matplotlib
import matplotlib.gridspec as gridspec
matplotlib.use("Agg") 
plt.style.use('/Users/krd020/Documents/qbo-metrics/plot_style.mplstyle')



from qbo_metrics.composite import (compute_composite, extract_events)
from qbo_metrics.metrics import (cycle_coherence, daily_composite_period, descent_rate,
                     latitudinal_width, max_min_amplitude, phase_amplitude,
                     zero_crossing_period_range)
from qbo_metrics.plotting import (plot_composite, plot_cycle_coherence,
                      plot_cycle_summary, plot_descent_rate, plot_period_fits, plot_phase_amplitude)
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


def _plot_reference_pressure(ax, pressure=30.0, label=None, linewidth=1.0):
    """Add the reference pressure to a pressure-axis plot."""
    if pressure is None:
        return None
    return ax.axhline(float(pressure), color="0.3", lw=linewidth, ls="--",
                      alpha=0.8, label=label)


def plot_composite(da, x, title="", x_label="", overlay=None,
                   overlay_step=None, overlay_color="k",
                   overlay_linestyles=None, pres_range=(1, 200), ax=None,
                   reference_pres=30.0, cb_label=None, levels=None):
    """Plot a pressure--``x`` composite as filled contours.

    Fill levels use a zero-centred diverging scale. ``overlay`` adds labelled
    line contours at ``overlay_step`` intervals. Pressure defaults to
    1--200 hPa; ``ax`` selects an existing Matplotlib axes. A dashed line
    marks ``reference_pres`` when it is not ``None``.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    else:
        fig = ax.figure
    if pres_range is not None:
        da = da.sel(pres=slice(*sorted(pres_range)))
        if overlay is not None:
            overlay = overlay.sel(pres=da["pres"], method='nearest')
    lim = float(abs(da).max())
    if levels is None:
        levels = ticker.MaxNLocator(nbins=20, symmetric=True).tick_values(-lim, lim)
    
    cs = ax.contourf(da[x], da["pres"], da.transpose("pres", x), levels=levels, cmap="RdBu_r")
    if overlay is not None:
        if overlay_step is None:
            raise ValueError("overlay_step must be given with overlay")
        olevels = ticker.MultipleLocator(overlay_step).tick_values(float(overlay.min()), float(overlay.max()))
        widths = np.where(olevels == 0, 1.8, 0.8)
        cl = ax.contour(overlay[x], overlay["pres"], overlay.transpose("pres", x),
                        levels=olevels, colors=overlay_color, linewidths=widths, linestyles=overlay_linestyles)
        ax.clabel(cl, fontsize=8, fmt="%g")
    ax.set_yscale("log")
    if pres_range is None:
        ax.invert_yaxis()
    else:
        ax.set_ylim(max(pres_range), min(pres_range))
    _plot_reference_pressure(ax, reference_pres)
    if x == "lag":
        ax.axvline(0, color="0.25", lw=0.8, ls="--", alpha=0.7)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Pressure [hPa]")
    ax.set_title(title)
    fig.colorbar(cs, ax=ax, label=cb_label)
    return fig


def _expand_shear_fits(fits):
    """Expand shear-resolved results into labelled plotting rows."""
    expanded = []
    for name, ds in fits.items():
        if "shear" in ds.dims:
            expanded.extend((f"{name} ({shear})", ds.sel(shear=shear))
                            for shear in ds["shear"].values)
        else:
            expanded.append((name, ds))
    return expanded


def plot_period_fits(fits, ax=None, y_label=None):
    """Plot reference-pressure profiles and their sinusoidal fits.

    ``fits`` maps labels to ``phase_amplitude`` results. Each panel reports
    period, half-range amplitude, sine-fit amplitude and phase.
    """
    expanded = _expand_shear_fits(fits)
    var, ds = expanded[0]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    else:
        fig = ax.figure

    reference_pres = ds.attrs.get("reference_pres", 30.0)
    f = (ds.sel(pres=reference_pres) if reference_pres in ds["pres"]
            else ds.interp(pres=reference_pres))
    period = float(ds["period"])

    ax.grid(True)
    ax.axhline(0, color="0.85", lw=1)
    ax.axvline(0, color="0.25", lw=0.8, ls="--", alpha=0.7)
    ax.plot(f["lag"], f["profile"], color="red", lw=2,
            label="Composite")
    ax.plot(f["lag"], f["fit"], color="blue", lw=2, label="Sine fit")
    ax.set_xlabel("Lag [months]")
    if y_label is None:
        ax.set_ylabel(f"{var} anomaly at 30 hPa")
    else:
        ax.set_ylabel(y_label)
    period_text = f"Period: {period:.1f} months"
    if "period_by_composite" in ds:
        components = ", ".join(
            f"{name}: {float(value):.1f}"
            for name, value in zip(ds["period_composite"].values,
                                    ds["period_by_composite"].values))
        period_text += f" (mean of {components})"
    ax.set_title(f"{period_text}")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, ncols=1, frameon=False, loc='lower center', bbox_to_anchor=(0.5, -0.7))
    return fig


def _wrapped_phase_line(phase, pres, phase_limits=(-np.pi, np.pi)):
    """Return a phase line wrapped to ``phase_limits`` with edge breaks."""
    phase = np.asarray(phase, dtype=float)
    pres = np.asarray(pres, dtype=float)
    lower, upper = phase_limits
    width = upper - lower
    if not np.isfinite(lower + upper) or width < 2 * np.pi - 1e-10:
        raise ValueError("phase_limits must span at least 2*pi radians")

    unwrapped = np.full_like(phase, np.nan)
    finite = np.isfinite(phase)
    starts = np.flatnonzero(finite & np.r_[True, ~finite[:-1]])
    ends = np.flatnonzero(finite & np.r_[~finite[1:], True]) + 1
    for start, end in zip(starts, ends):
        unwrapped[start:end] = np.unwrap(phase[start:end])
    if finite.any():
        centre = (lower + upper) / 2
        unwrapped[finite] += 2 * np.pi * np.round(
            (centre - np.nanmedian(unwrapped)) / (2 * np.pi))

    marker_phase = lower + np.mod(unwrapped - lower, width)
    first = int(np.flatnonzero(finite)[0]) if finite.any() else 0
    plot_phase = [marker_phase[first]] if finite.any() else [np.nan]
    plot_pres = [pres[first]] if finite.any() else [np.nan]

    for phase0, phase1, pres0, pres1 in zip(unwrapped[:-1], unwrapped[1:],
                                            pres[:-1], pres[1:]):
        if not np.all(np.isfinite((phase0, phase1, pres0, pres1))):
            plot_phase.extend((np.nan, phase1))
            plot_pres.extend((np.nan, pres1))
            continue

        phase0_plot = lower + np.mod(phase0 - lower, width)
        phase1_plot = phase0_plot + phase1 - phase0
        if phase1_plot > upper:
            edge, opposite_edge = upper, lower
            phase1_wrapped = phase1_plot - width
        elif phase1_plot < lower:
            edge, opposite_edge = lower, upper
            phase1_wrapped = phase1_plot + width
        else:
            plot_phase.append(phase1_plot)
            plot_pres.append(pres1)
            continue

        weight = (edge - phase0_plot) / (phase1_plot - phase0_plot)
        crossing_pres = np.exp(np.log(pres0) + weight *
                               (np.log(pres1) - np.log(pres0)))
        plot_phase.extend((edge, np.nan, opposite_edge, phase1_wrapped))
        plot_pres.extend((crossing_pres, np.nan, crossing_pres, pres1))

    return np.asarray(plot_phase), np.asarray(plot_pres), marker_phase


def _vertical_extent_specs(source):
    """Return finite top and bottom extent values with plot metadata."""
    if source is None:
        return []
    specs = []
    for boundary, name, linestyle in (
            ("top", "vertical_extent_top", "-."),
            ("bottom", "vertical_extent_bottom", ":")):
        if name not in source:
            continue
        data = source[name]
        value = float(data)
        if not np.isfinite(value):
            continue
        fraction = getattr(data, "attrs", {}).get("fraction", 0.1)
        specs.append({
            "boundary": boundary,
            "value": value,
            "fraction": fraction,
            "linestyle": linestyle,
            "label": f"{boundary.capitalize()} {fraction:.0%} extent",
        })
    return specs


def plot_phase_amplitude(fits, title="QBO phase and amplitude",
                         show_vertical_extent=True,
                         phase_limits=(-np.pi, np.pi),
                         pres_range=(1, 200),
                         amp_ax=None, phase_ax=None):
    """Plot half-range amplitude, sine-fit amplitude and phase profiles.

    ``fits`` maps labels to ``phase_amplitude`` results. Each row shows
    half-range amplitude, sine-fit amplitude and wrapped phase.
    ``show_vertical_extent`` marks the top and bottom sine-fit amplitude
    thresholds; ``phase_limits`` controls phase wrapping.

    ``amp_ax`` and ``phase_ax`` optionally supply existing axes to draw
    into. Provide both or neither. Each may be a single Axes (only if
    ``fits`` expands to one row) or a sequence of Axes, one per row. If
    omitted, a new figure is created. The figure title is only set on
    figures created here.
    """
    expanded = _expand_shear_fits(fits)
    n_rows = len(expanded)

    if (amp_ax is None) != (phase_ax is None):
        raise ValueError("Provide both amp_ax and phase_ax, or neither.")

    if amp_ax is None:
        fig, axes = plt.subplots(n_rows, 2, squeeze=False,
                                 figsize=(9, 3.5 * n_rows),
                                 constrained_layout=True, sharey="row")
        amp_axes, phase_axes = list(axes[:, 0]), list(axes[:, 1])
        created_fig = True
    else:
        amp_axes = list(np.atleast_1d(amp_ax).ravel())
        phase_axes = list(np.atleast_1d(phase_ax).ravel())
        if len(amp_axes) != n_rows or len(phase_axes) != n_rows:
            raise ValueError(
                f"Expected {n_rows} axes in each of amp_ax and phase_ax "
                f"(one per row); got {len(amp_axes)} and {len(phase_axes)}.")
        fig = amp_axes[0].figure
        created_fig = False

    lower, upper = phase_limits
    half_pi = np.pi / 2
    phase_ticks = np.arange(np.ceil(lower / half_pi),
                            np.floor(upper / half_pi) + 1) * half_pi

    def pi_label(value):
        multiple = int(np.round(value / half_pi))
        if multiple == 0:
            return "0"
        sign = "-" if multiple < 0 else ""
        numerator = abs(multiple)
        if numerator % 2 == 0:
            coefficient = numerator // 2
            return rf"${sign}\pi$" if coefficient == 1 else rf"${sign}{coefficient}\pi$"
        return rf"${sign}\pi/2$" if numerator == 1 else rf"${sign}{numerator}\pi/2$"

    phase_labels = [pi_label(value) for value in phase_ticks]

    for (var, ds), amp_ax_i, phase_ax_i in zip(expanded, amp_axes,
                                               phase_axes):
        units = ds["profile"].attrs.get("units", "")
        units = f" ({units})" if units else ""
        reference_pres = ds.attrs.get("reference_pres", 30.0)

        amp_ax_i.plot(ds["amp"], ds["pres"], color="b", marker="o",
                      ms=3, label="Sine-fit amplitude")
        if "amplitude" in ds:
            amp_ax_i.plot(ds["amplitude"], ds["pres"], color="purple",
                          marker="o", ms=3, ls="--",
                          label="Half-range amplitude")
        _plot_reference_pressure(
            amp_ax_i, reference_pres,
            label=f"Reference ({reference_pres:g} hPa)")
        amp_ax_i.set_xlabel(f"Amplitude{units}")
        amp_ax_i.set_ylabel("Pressure [hPa]")
        amp_ax_i.set_title("Amplitude")

        phase_ax_i.axvline(0, color="0.85", lw=1)
        phase_line, phase_pres, phase_markers = _wrapped_phase_line(
            ds["phase"], ds["pres"], phase_limits=phase_limits)
        phase_ax_i.plot(phase_line, phase_pres, color="r")
        phase_ax_i.plot(phase_markers, ds["pres"], color="r", ls="none",
                        marker="o", ms=3)
        _plot_reference_pressure(phase_ax_i, reference_pres)
        extent_specs = (_vertical_extent_specs(ds)
                        if show_vertical_extent else [])
        for spec in extent_specs:
            amp_ax_i.axhline(spec["value"], color="green", lw=1.5,
                             ls=spec["linestyle"], label=spec["label"])
            #amp_ax_i.annotate(
            #    f"{spec['value']:.3g} hPa",
            #    xy=(0.99, spec["value"]), xycoords=("axes fraction", "data"),
            #    xytext=(0, -2),
            #    textcoords="offset points", ha="right",
            #    va="top", color="0.4", fontsize=8)
            phase_ax_i.axhline(spec["value"], color="green", lw=1.5,
                               ls=spec["linestyle"])
        phase_ax_i.set_xlim(*phase_limits)
        phase_ax_i.set_xticks(phase_ticks, phase_labels)
        phase_ax_i.set_xlabel("Phase (radians)")
        phase_ax_i.set_title("Phase")
        phase_ax_i.set_ylabel("Pressure (hPa)")

        for ax in (amp_ax_i, phase_ax_i):
            ax.set_yscale("log")
            ax.grid(color="0.9", lw=0.8)

        # User-supplied axes may not share a y-axis, so only rely on
        # sharing when it is actually in place.
        y_shared = amp_ax_i.get_shared_y_axes().joined(amp_ax_i, phase_ax_i)
        if pres_range is None:
            amp_ax_i.invert_yaxis()
            if not y_shared:
                phase_ax_i.invert_yaxis()
        else:
            top, bottom = sorted(pres_range)
            for spec in extent_specs:
                if spec["boundary"] == "top":
                    top = min(top, spec["value"])
                    if spec["value"] < min(pres_range):
                        top = min(top, float(ds["pres"].min()))
                else:
                    bottom = max(bottom, spec["value"])
                    if spec["value"] > max(pres_range):
                        bottom = max(bottom, float(ds["pres"].max()))
            amp_ax_i.set_ylim(bottom, top)
            if not y_shared:
                phase_ax_i.set_ylim(bottom, top)

    legend_items = {}
    for ax in amp_axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            legend_items.setdefault(label, handle)

    amp_ax_i.legend(legend_items.values(), legend_items.keys(),
               loc='lower center', bbox_to_anchor=(1.2, -0.8), ncols=2,
               frameon=False)
    if created_fig:
        fig.suptitle(title)
    return fig



if __name__ == "__main__":

    dates = xr.open_dataset('/Users/krd020/Documents/qbo-metrics/reference_dates.nc')
    westerly_u = xr.open_dataset('/Users/krd020/Documents/qbo-metrics/qbo_composite_westerly_exclude_2015_2020.nc')
    period_range = zero_crossing_period_range(dates, exclude_years=EXCLUDED_ONSET_YEARS)

    ace_loc = "/Volumes/Users/krd020/postdoc/ISSI_QBO/data/ACE_V5.3_O3_MZM_anom_2004-2026_for_qbo_metrics.nc" 
    ds_daily, ace_ds_monthly = load_data(ace_loc, variables=['o3'])
    ace_ds_monthly = ace_ds_monthly.load()
    ace_ds_monthly = ace_ds_monthly.sel(time=slice('2004-8', '2025-12'))
    ace_composites = _get_composites(ace_ds_monthly, dates)
    ace_westerly_composite = ace_composites['westerly'].load()
    ace_easterly_composite = ace_composites['easterly'].load()
    ace_westerly_equatorial = ace_westerly_composite.sel(latitude=0)
    ace_easterly_equatorial = ace_easterly_composite.sel(latitude=0)
    ace_u_composites = {
        "westerly": ace_westerly_equatorial['o3'],
        "easterly": ace_easterly_equatorial['o3'],
    }
    ace_u_eq_monthly = (ace_ds_monthly.sel(pres=slice(0.5, 200), latitude=[0]).mean("latitude").load())
    ace_u_cycles = phase_amplitude(ace_u_composites, zero_crossing_range=period_range)
    ace_qbo_period = float(ace_u_cycles["period"])
    ace_cycle_fits = {
        'o3': phase_amplitude(
            ace_westerly_equatorial['o3'], period=ace_qbo_period,
            zero_crossing_range=period_range),}


    os_loc = "/Volumes/Users/krd020/postdoc/ISSI_QBO/data/OSIRIS_V7.4_O3_MZM_anom_2002-2021_for_qbo_metrics.nc" 
    ds_daily, os_ds_monthly = load_data(os_loc, variables=['ozone_concentration'])
    os_ds_monthly = os_ds_monthly.rename({'ozone_concentration': 'o3'})
    os_ds_monthly = os_ds_monthly.load()
    os_ds_monthly = os_ds_monthly.sel(time=slice('2004-8', '2025-12'))
    os_composites = _get_composites(os_ds_monthly, dates)
    os_westerly_composite = os_composites['westerly'].load()
    os_easterly_composite = os_composites['easterly'].load()
    os_westerly_equatorial = os_westerly_composite.sel(latitude=0)
    os_easterly_equatorial = os_easterly_composite.sel(latitude=0)
    os_u_composites = {
        "westerly": os_westerly_equatorial['o3'],
        "easterly": os_easterly_equatorial['o3'],
    }
    os_u_eq_monthly = (os_ds_monthly.sel(pres=slice(0.5, 200), latitude=[0]).mean("latitude").load())
    os_u_cycles = phase_amplitude(os_u_composites, zero_crossing_range=period_range)
    os_qbo_period = float(os_u_cycles["period"])
    os_cycle_fits = {
        'o3': phase_amplitude(
            os_westerly_equatorial['o3'], period=os_qbo_period,
            zero_crossing_range=period_range),}


    mls_loc = "/Volumes/Users/krd020/postdoc/ISSI_QBO/data/MLS_V6.0_O3_MZM_anom_2004-2026_for_qbo_metrics.nc"
    ds_daily, mls_ds_monthly = load_data(mls_loc, variables=['o3'])
    mls_ds_monthly = mls_ds_monthly.load()
    mls_ds_monthly =mls_ds_monthly.interp(pres=os_ds_monthly.pres)
    mls_ds_monthly = mls_ds_monthly.sel(time=slice('2004-8', '2025-12'))
    mls_composites = _get_composites(mls_ds_monthly, dates)
    mls_westerly_composite = mls_composites['westerly'].load()
    mls_easterly_composite = mls_composites['easterly'].load()
    mls_westerly_equatorial = mls_westerly_composite.sel(latitude=0)
    mls_easterly_equatorial = mls_easterly_composite.sel(latitude=0)
    mls_u_composites = {
        "westerly": mls_westerly_equatorial['o3'],
        "easterly": mls_easterly_equatorial['o3'],
    }
    mls_u_eq_monthly = (mls_ds_monthly.sel(pres=slice(0.5, 200), latitude=[0]).mean("latitude").load())
    mls_u_cycles = phase_amplitude(mls_u_composites, zero_crossing_range=period_range)
    mls_qbo_period = float(mls_u_cycles["period"])
    mls_cycle_fits = {
        'o3': phase_amplitude(
            mls_westerly_equatorial['o3'], period=mls_qbo_period,
            zero_crossing_range=period_range),}


    # plot
    # top row: difference between ace/osiris composites and mls
    # bottom row: comparison of period, amplitude, phase 


    fig = plt.figure(figsize=(7, 6))

    outer_gs = fig.add_gridspec(2, 1, hspace=0.6)
    top_gs = outer_gs[0].subgridspec(1, 2, wspace=0.5)
    bottom_gs = outer_gs[1].subgridspec(1, 3, wspace=0.5)
    ax00 = plt.subplot(top_gs[0, 0])
    plot_composite(
        westerly_equatorial[v], x="lag", x_label="Lag [months]",
        title="Westerly Composite", cb_label="O$_3$ anomaly [ppm]", ax=ax00)

    ax01 = plt.subplot(top_gs[0, 1])
    plot_composite(
        westerly_composite[v].sel(lag=0),
        x="latitude", x_label="Latitude [$^{\\circ}$]",
        title="Westerly Composite at lag=0", cb_label="%s anomaly [%s]" % (latex_names[vid], units[vid]),
        overlay=westerly_u["u"].sel(lag=0), overlay_step=10.0, ax=ax01)

    ax00 = plt.subplot(bottom_gs[0, 0])
    plot_period_fits(cycle_fits, ax=ax00, y_label="%s anomaly at 30 hPa [%s]" % (latex_names[vid], units[vid]))

    ax01 = plt.subplot(bottom_gs[0, 1])
    ax02 = plt.subplot(bottom_gs[0, 2])
    plot_phase_amplitude(cycle_fits, amp_ax=ax01, phase_ax=ax02)

    plt.subplots_adjust(bottom=0.2)
    plt.savefig('/Volumes/Users/krd020/postdoc/ISSI_QBO/figs/MLS_OSIRIS_ACE_ozone_comp.png')
