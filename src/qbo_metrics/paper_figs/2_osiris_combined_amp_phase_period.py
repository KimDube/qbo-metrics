import numpy as np
import xarray as xr
from matplotlib import pyplot as plt, ticker
import matplotlib
matplotlib.use("Agg") 
plt.style.use('/Users/krd020/Documents/qbo-metrics/plot_style.mplstyle')
from qbo_metrics.utils import load_control_file




def _plot_reference_pressure(ax, pressure=30.0, label=None, linewidth=1.0):
    """Add the reference pressure to a pressure-axis plot."""
    if pressure is None:
        return None
    return ax.axhline(float(pressure), color="0.3", lw=linewidth, ls="--",
                      alpha=0.8, label=label)


def plot_composite(da, x, title="", x_label="", overlay=None,
                   overlay_step=None, overlay_color="k",
                   overlay_linestyles=None, pres_range=(1, 200), ax=None,
                   reference_pres=30.0, levels=None, y_label='Pressure [hPa]'):
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
    
    cs = ax.contourf(da[x], da["pres"], da.transpose("pres", x), levels=levels, cmap="RdBu_r", extend='both')
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
    ax.set_ylabel(y_label)
    ax.set_title(title)
    return cs


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
    ax.legend(handles, labels, ncols=1, frameon=False, loc='lower center', bbox_to_anchor=(0.5, -0.6))
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
        _plot_reference_pressure(amp_ax_i, reference_pres)
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

        phase_ax_i.set_xlim(*phase_limits)
        phase_ax_i.set_xticks(phase_ticks, phase_labels)
        phase_ax_i.set_xlabel("Phase [radians]")
        phase_ax_i.set_title("Phase")
        phase_ax_i.set_yticklabels("")

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
               loc='lower center', bbox_to_anchor=(0.6, -0.6), ncols=1,
               frameon=False)
    if created_fig:
        fig.suptitle(title)
    phase_ax_i.set_yticklabels("")
    phase_ax_i.set_ylim([200, 1])
    amp_ax_i.set_ylim([200, 1])
    return fig


def plot_descent_rate(result, title="Zero-wind descent rate",
                      reference_pres=30.0, ax=None):
    """Plot composite descent-rate profiles and cycle-to-cycle spread."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    else:
        fig = ax.figure
    colours = {"easterly": "blue", "westerly": "red"}
    pressure = np.asarray(result["pres"].values, dtype=float)
    plotted = False
    for shear in result["shear"].values:
        colour = colours[str(shear)]
        profile = result.sel(shear=shear)
        rate = np.asarray(profile["descent_rate"].values, dtype=float)
        std = np.asarray(profile["std_descent_rate"].values, dtype=float)
        valid = np.isfinite(pressure) & np.isfinite(rate)
        plotted |= valid.any()
        label = (f"{str(shear).capitalize()} onset")
        ax.plot(rate, pressure, color=colour, lw=2, label=label)
        spread_valid = valid & np.isfinite(std)
        ax.fill_betweenx(pressure, rate - std, rate + std,
                         where=spread_valid, color=colour, alpha=0.15)
    if not plotted:
        plt.close(fig)
        raise ValueError("No finite descent-rate profile values to plot")
    ax.axvline(0, color="0.7", lw=0.8)
    ax.set_xlabel("Downward rate\n [hPa month$^{-1}$]")
    ax.set_yscale("log")
    ax.set_ylim(float(np.nanmax(pressure)), float(np.nanmin(pressure)))
    ax.set_xlim([-10, 30])
    _plot_reference_pressure(ax, reference_pres)
    ax.set_yticklabels("")
    ax.set_title(title)
    ax.grid(color="0.9", lw=0.8)
    ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, -0.6), ncols=1)
    ax.set_ylim([200, 1])
    return fig


if __name__ == "__main__":

    SETTINGS = load_control_file()
    EXCLUDED_ONSET_YEARS = SETTINGS['EXCLUDED_ONSET_YEARS']
    SHEAR_TYPES = SETTINGS['SHEAR_TYPES']
    DATA_LOC = SETTINGS['data_save_path']
    FIG_LOC = SETTINGS['fig_save_path']
    ROOT_LOC = SETTINGS['root_path']

    dates = xr.open_dataset(ROOT_LOC + 'reference_dates.nc')
    westerly_u = xr.open_dataset(ROOT_LOC + '/qbo_composite_westerly_exclude_2015_2020.nc')
    easterly_u = xr.open_dataset(ROOT_LOC + 'qbo_composite_easterly_exclude_2015_2020.nc')

    latex_names = SETTINGS['OSIRIS_latex_names']
    units = SETTINGS['OSIRIS_units']

    for vid, var in enumerate(SETTINGS['OSIRIS_VARS']):  # 
        data_loc = DATA_LOC + "OSIRIS_V7.4_%s_MZM_anom_2004-2026_for_qbo_metrics.nc" % var
        v = var.lower()

        westerly_composite = xr.open_dataset(DATA_LOC + 'OSIRIS_V7.4_%s_westerly_composite.nc' % var)
        westerly_equatorial = westerly_composite.sel(latitude=0)
        cycles = xr.open_dataset(DATA_LOC + 'OSIRIS_V7.4_%s_westerly_eq_cyles.nc' % var)
        cycle_fits = {v: cycles,}
        descent = xr.open_dataset(DATA_LOC + 'OSIRIS_V7.4_%s_descent.nc' % var)

        fig = plt.figure(figsize=(7, 6))
        fig.suptitle("OSIRIS %s" % latex_names[vid], y=0.93)

        outer = fig.add_gridspec(2, 1, hspace=0.45)
        # Top row: [two axes | colorbar], all inside the same width as the bottom row
        top = outer[0].subgridspec(1, 2, width_ratios=[1, 0.03], wspace=0.03)
        top_axes = top[0].subgridspec(1, 2, wspace=0.1)
        top_left = fig.add_subplot(top_axes[0])
        top_right = fig.add_subplot(top_axes[1])
        cax = fig.add_subplot(top[1])

        # Bottom row: solo axis (1/4 width) + group of three (3/4 width)
        bottom = outer[1].subgridspec(1, 2, width_ratios=[1, 3], wspace=0.3)
        solo = fig.add_subplot(bottom[0])

        group = bottom[1].subgridspec(1, 3, wspace=0.1)
        b1 = fig.add_subplot(group[0])
        b2 = fig.add_subplot(group[1])
        b3 = fig.add_subplot(group[2])

        cblevels = np.arange(SETTINGS['OSIRIS_vmin'][vid], SETTINGS['OSIRIS_vmax'][vid], SETTINGS['OSIRIS_freq'][vid])
        plot_composite(
            westerly_equatorial[v], x="lag", x_label="Lag [months]",
            title="Westerly Composite",
            overlay=westerly_u["u"].sel(latitude=0), overlay_step=10.0, ax=top_left, levels=cblevels)

        cs = plot_composite(
            westerly_composite[v].sel(lag=0),
            x="latitude", x_label="Latitude [$^{\\circ}$]",
            title="Westerly Composite at lag=0",
            overlay=westerly_u["u"].sel(lag=0), overlay_step=10.0, ax=top_right, y_label=None, levels=cblevels)
        top_right.set_yticklabels("")
        cbar = fig.colorbar(cs, cax=cax)
        cbar.set_label("%s anomaly [%s]" % (latex_names[vid], units[vid]))

        plot_period_fits(cycle_fits, ax=solo, y_label="%s anomaly at 30 hPa [%s]" % (latex_names[vid], units[vid]))
        plot_phase_amplitude(cycle_fits, amp_ax=b1, phase_ax=b2)
        plot_descent_rate(descent, ax=b3)

        top_left.text(0.03, 0.9, '(a)', transform=top_left.transAxes, fontsize=9)
        top_right.text(0.03, 0.9, '(b)', transform=top_right.transAxes, fontsize=9)
        solo.text(0.03, 0.9, '(c)', transform=solo.transAxes, fontsize=9)
        b1.text(0.8, 0.9, '(d)', transform=b1.transAxes, fontsize=9)
        b2.text(0.03, 0.9, '(e)', transform=b2.transAxes, fontsize=9)
        b3.text(0.03, 0.9, '(f)', transform=b3.transAxes, fontsize=9)

        plt.subplots_adjust(bottom=0.2)
        plt.savefig('/Volumes/Users/krd020/postdoc/ISSI_QBO/figs/OSIRIS_V7.4_%s_westerly_amp_phase_period.png' % var)
