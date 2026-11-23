import numpy as np
import xarray as xr
from matplotlib import pyplot as plt, ticker
import matplotlib
matplotlib.use("Agg") 
plt.style.use('/Users/krd020/Documents/qbo-metrics/plot_style.mplstyle')

from qbo_metrics.metrics import zero_crossing_period_range
from qbo_metrics.utils import load_control_file



def _plot_reference_pressure(ax, pressure=30.0, label=None, linewidth=1.0):
    """Add the reference pressure to a pressure-axis plot."""
    if pressure is None:
        return None
    return ax.axhline(float(pressure), color="0.3", lw=linewidth, ls="--",
                      alpha=0.8, label=label)


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




if __name__ == "__main__":

    SETTINGS = load_control_file()
    EXCLUDED_ONSET_YEARS = SETTINGS['EXCLUDED_ONSET_YEARS']
    SHEAR_TYPES = SETTINGS['SHEAR_TYPES']
    DATA_LOC = SETTINGS['data_save_path']
    FIG_LOC = SETTINGS['fig_save_path']
    ROOT_LOC = SETTINGS['root_path']

    dates = xr.open_dataset(ROOT_LOC + 'reference_dates.nc')
    westerly_u = xr.open_dataset(ROOT_LOC + '/qbo_composite_westerly_exclude_2015_2020.nc')
    period_range = zero_crossing_period_range(dates, exclude_years=EXCLUDED_ONSET_YEARS)

    ace_westerly_composite = xr.open_dataset(DATA_LOC + 'ACE_V5.3_O3_westerly_composite.nc')
    ace_westerly_equatorial = ace_westerly_composite.sel(latitude=0)
    ace_cycles = xr.open_dataset(DATA_LOC + 'ACE_V5.3_O3_westerly_eq_cyles.nc')
    ace_cycle_fits = {'o3': ace_cycles,}
    ace_descent = xr.open_dataset(DATA_LOC + 'ACE_V5.3_O3_descent.nc')

    os_westerly_composite = xr.open_dataset(DATA_LOC + 'OSIRIS_V7.4_O3_westerly_composite.nc')
    os_westerly_equatorial = os_westerly_composite.sel(latitude=0)
    os_cycles = xr.open_dataset(DATA_LOC + 'OSIRIS_V7.4_O3_westerly_eq_cyles.nc')
    os_cycle_fits = {'o3': os_cycles,}
    os_descent = xr.open_dataset(DATA_LOC + 'OSIRIS_V7.4_O3_descent.nc')

    mls_westerly_composite = xr.open_dataset(DATA_LOC + 'MLS_V6.0_O3_westerly_composite.nc')
    mls_westerly_equatorial = mls_westerly_composite.sel(latitude=0)
    mls_cycles = xr.open_dataset(DATA_LOC + 'MLS_V6.0_O3_westerly_eq_cyles.nc')
    mls_cycle_fits = {'o3': mls_cycles,}
    mls_descent = xr.open_dataset(DATA_LOC + 'MLS_V6.0_O3_descent.nc')

    # plot
    # top row: difference between ace/osiris composites and mls
    # bottom row: comparison of period, amplitude, phase, descent
    reference_pres = 30.0

    fig = plt.figure(figsize=(8, 6))
    fig.suptitle("O$_3$", y=0.95)
    outer_gs = fig.add_gridspec(2, 1, hspace=0.6)
    top_gs = outer_gs[0].subgridspec(1, 3, wspace=0.8)
    bottom_gs = outer_gs[1].subgridspec(1, 4, wspace=0.2)

    ax00 = plt.subplot(top_gs[0, 0])
    diff = mls_westerly_composite.interp(pres=os_westerly_composite.pres).sel(lag=0).o3 - os_westerly_composite.sel(lag=0).o3
    cs = ax00.contourf(diff.latitude, diff.pres, diff, levels=np.arange(-0.3, 0.305, 0.05), cmap="PiYG_r", extend='both')
    ax00.set_yscale("log")
    ax00.invert_yaxis()
    _plot_reference_pressure(ax00, reference_pres)
    ax00.set_xlabel("Latitude [$^{\\circ}$]")
    ax00.set_ylabel("Pressure [hPa]")
    ax00.set_title("Westerly composite\n lag=0")
    fig.colorbar(cs, ax=ax00, label="MLS$-$OSIRIS [ppm]")

    ax01 = plt.subplot(top_gs[0, 1])
    diff = mls_westerly_composite.interp(pres=os_westerly_composite.pres).sel(lag=0).o3 - ace_westerly_composite.sel(lag=0).o3
    cs = ax01.contourf(diff.latitude, diff.pres, diff, levels=np.arange(-0.3, 0.305, 0.05), cmap="PiYG_r", extend='both')
    ax01.set_yscale("log")
    ax01.invert_yaxis()
    _plot_reference_pressure(ax01, reference_pres)
    ax01.set_xlabel("Latitude [$^{\\circ}$]")
    ax01.set_ylabel("Pressure [hPa]")
    ax01.set_title("Westerly composite\n lag=0")
    fig.colorbar(cs, ax=ax01, label="MLS$-$ACE-FTS [ppm]")

    ax00 = plt.subplot(top_gs[0, 2])
    ax00.grid(True)
    ax00.axhline(0, color="0.85", lw=1)
    ax00.axvline(0, color="0.25", lw=0.8, ls="--", alpha=0.7)
    expanded = _expand_shear_fits(mls_cycle_fits)
    var, ds = expanded[0]
    f = (ds.sel(pres=reference_pres) if reference_pres in ds["pres"]
            else ds.interp(pres=reference_pres))
    period = float(ds["period"])
    ax00.plot(f["lag"], f["profile"], color="red", label="MLS (T=%.1f months)" % period)
    expanded = _expand_shear_fits(os_cycle_fits)
    var, ds = expanded[0]
    f = (ds.sel(pres=reference_pres) if reference_pres in ds["pres"]
            else ds.interp(pres=reference_pres))
    period = float(ds["period"])
    ax00.plot(f["lag"], f["profile"], color="blue", label="OSIRIS (T=%.1f months)" % period)
    expanded = _expand_shear_fits(ace_cycle_fits)
    var, ds = expanded[0]
    f = (ds.sel(pres=reference_pres) if reference_pres in ds["pres"]
            else ds.interp(pres=reference_pres))
    period = float(ds["period"])
    ax00.plot(f["lag"], f["profile"], color="orange", label="ACE-FTS (T=%.1f months)" % period)

    ax00.set_title("Period")
    ax00.set_xlabel("Lag [months]")
    ax00.set_ylabel(f"O$_3$ anomaly at 30 hPa [ppm]")
    handles, labels = ax00.get_legend_handles_labels()


    ax01 = plt.subplot(bottom_gs[0, 0])

    expanded = _expand_shear_fits(mls_cycle_fits)
    var, ds = expanded[0]
    ax01.plot(ds["amplitude"], ds["pres"], color="red")
    expanded = _expand_shear_fits(os_cycle_fits)
    var, ds = expanded[0]
    ax01.plot(ds["amplitude"], ds["pres"], color="blue")
    expanded = _expand_shear_fits(ace_cycle_fits)
    var, ds = expanded[0]
    ax01.plot(ds["amplitude"], ds["pres"], color="orange")
    ax01.invert_yaxis()
    ax01.set_yscale("log")
    ax01.grid(True)
    ax01.set_title("Half-range amplitude")
    ax01.set_xlabel("Amplitude [ppm]")
    ax01.set_ylabel("Pressure [hPa]")
    ax01.set_ylim([200, 1])
    _plot_reference_pressure(ax01, reference_pres)
    ax01.legend(handles, labels, ncols=3, frameon=False, loc='lower center', bbox_to_anchor=(2.2, -0.6))


    ax02 = plt.subplot(bottom_gs[0, 1])
    phase_limits=(-np.pi, np.pi)

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


    expanded = _expand_shear_fits(mls_cycle_fits)
    var, ds = expanded[0]
    phase_line, phase_pres, phase_markers = _wrapped_phase_line(
        ds["phase"], ds["pres"], phase_limits=phase_limits)
    ax02.plot(phase_line, phase_pres, color="r")

    expanded = _expand_shear_fits(os_cycle_fits)
    var, ds = expanded[0]
    phase_line, phase_pres, phase_markers = _wrapped_phase_line(
        ds["phase"], ds["pres"], phase_limits=phase_limits)
    ax02.plot(phase_line, phase_pres, color="blue")

    expanded = _expand_shear_fits(ace_cycle_fits)
    var, ds = expanded[0]
    phase_line, phase_pres, phase_markers = _wrapped_phase_line(
        ds["phase"], ds["pres"], phase_limits=phase_limits)
    ax02.plot(phase_line, phase_pres, color="orange")
    _plot_reference_pressure(ax02, reference_pres)
    ax02.set_xlim(*phase_limits)
    ax02.set_xticks(phase_ticks, phase_labels)
    ax02.invert_yaxis()
    ax02.set_yscale("log")
    ax02.grid(True)
    ax02.set_title("Phase")
    ax02.set_xlabel("Phase [radians]")
    ax02.set_yticklabels([])
    ax02.set_ylim([200, 1])


    ax03 = plt.subplot(bottom_gs[0, 2])
    pressure = np.asarray(mls_descent["pres"].values, dtype=float)
    shear = 'westerly'
    profile = mls_descent.sel(shear=shear)
    rate = np.asarray(profile["descent_rate"].values, dtype=float)
    std = np.asarray(profile["std_descent_rate"].values, dtype=float)
    valid = np.isfinite(pressure) & np.isfinite(rate)
    label = (f"{str(shear).capitalize()} onset "
                f"({float(profile['mean_descent_rate']):.2f} "
                "hPa month$^{-1}$)")
    ax03.plot(rate, pressure, color='red', label=label)
    spread_valid = valid & np.isfinite(std)
    ax03.fill_betweenx(pressure, rate - std, rate + std,
                        where=spread_valid, color='red', alpha=0.15)

    pressure = np.asarray(os_descent["pres"].values, dtype=float)
    profile = os_descent.sel(shear=shear)
    rate = np.asarray(profile["descent_rate"].values, dtype=float)
    std = np.asarray(profile["std_descent_rate"].values, dtype=float)
    valid = np.isfinite(pressure) & np.isfinite(rate)
    label = (f"{str(shear).capitalize()} onset "
                f"({float(profile['mean_descent_rate']):.2f} "
                "hPa month$^{-1}$)")
    ax03.plot(rate, pressure, color='blue', label=label)
    spread_valid = valid & np.isfinite(std)
    ax03.fill_betweenx(pressure, rate - std, rate + std,
                        where=spread_valid, color='blue', alpha=0.15)

    pressure = np.asarray(ace_descent["pres"].values, dtype=float)
    profile = ace_descent.sel(shear=shear)
    rate = np.asarray(profile["descent_rate"].values, dtype=float)
    std = np.asarray(profile["std_descent_rate"].values, dtype=float)
    valid = np.isfinite(pressure) & np.isfinite(rate)
    label = (f"{str(shear).capitalize()} onset "
                f"({float(profile['mean_descent_rate']):.2f} "
                "hPa month$^{-1}$)")
    ax03.plot(rate, pressure, color='orange', label=label)
    spread_valid = valid & np.isfinite(std)
    ax03.fill_betweenx(pressure, rate - std, rate + std,
                        where=spread_valid, color='orange', alpha=0.15)

    ax03.axvline(0, color="0.7", lw=0.8)
    ax03.set_xlabel("Downward rate\n [hPa month$^{-1}$]")
    ax03.set_yscale("log")
    ax03.set_ylim(float(np.nanmax(pressure)), float(np.nanmin(pressure)))
    _plot_reference_pressure(ax03, reference_pres)
    ax03.set_title("Westerly Onset")
    ax03.grid(True)
    ax03.set_xlim([-5, 40])
    ax03.set_yticklabels([])

    ax03 = plt.subplot(bottom_gs[0, 3])
    shear = 'easterly'

    pressure = np.asarray(mls_descent["pres"].values, dtype=float)
    profile = mls_descent.sel(shear=shear)
    rate = np.asarray(profile["descent_rate"].values, dtype=float)
    std = np.asarray(profile["std_descent_rate"].values, dtype=float)
    valid = np.isfinite(pressure) & np.isfinite(rate)
    label = (f"{str(shear).capitalize()} onset "
                f"({float(profile['mean_descent_rate']):.2f} "
                "hPa month$^{-1}$)")
    ax03.plot(rate, pressure, color='red', label=label)
    spread_valid = valid & np.isfinite(std)
    ax03.fill_betweenx(pressure, rate - std, rate + std,
                        where=spread_valid, color='red', alpha=0.15)

    pressure = np.asarray(os_descent["pres"].values, dtype=float)
    profile = os_descent.sel(shear=shear)
    rate = np.asarray(profile["descent_rate"].values, dtype=float)
    std = np.asarray(profile["std_descent_rate"].values, dtype=float)
    valid = np.isfinite(pressure) & np.isfinite(rate)
    label = (f"{str(shear).capitalize()} onset "
                f"({float(profile['mean_descent_rate']):.2f} "
                "hPa month$^{-1}$)")
    ax03.plot(rate, pressure, color='blue',  label=label)
    spread_valid = valid & np.isfinite(std)
    ax03.fill_betweenx(pressure, rate - std, rate + std,
                        where=spread_valid, color='blue', alpha=0.15)

    pressure = np.asarray(ace_descent["pres"].values, dtype=float)
    profile = ace_descent.sel(shear=shear)
    rate = np.asarray(profile["descent_rate"].values, dtype=float)
    std = np.asarray(profile["std_descent_rate"].values, dtype=float)
    valid = np.isfinite(pressure) & np.isfinite(rate)
    label = (f"{str(shear).capitalize()} onset "
                f"({float(profile['mean_descent_rate']):.2f} "
                "hPa month$^{-1}$)")
    ax03.plot(rate, pressure, color='orange', label=label)
    spread_valid = valid & np.isfinite(std)
    ax03.fill_betweenx(pressure, rate - std, rate + std,
                        where=spread_valid, color='orange', alpha=0.15)

    ax03.axvline(0, color="0.7", lw=0.8)
    ax03.set_xlabel("Downward rate\n [hPa month$^{-1}$]")
    ax03.set_yscale("log")
    ax03.set_ylim(float(np.nanmax(pressure)), float(np.nanmin(pressure)))
    _plot_reference_pressure(ax03, reference_pres)
    ax03.set_title("Easterly Onset")
    ax03.grid(True)
    ax03.set_xlim([-5, 15])
    ax03.set_yticklabels([])

    plt.subplots_adjust(bottom=0.2)
    plt.savefig('/Volumes/Users/krd020/postdoc/ISSI_QBO/figs/MLS_OSIRIS_ACE_ozone_comp.png')
