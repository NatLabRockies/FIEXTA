import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import numpy as np
from lidargo import utilities
import warnings

# Suppress all graphical UserWarnings
warnings.filterwarnings(
    "ignore",
    message=".*interpreted as cell centers, but are not monotonically increasing or decreasing.*",
)

warnings.filterwarnings(
    "ignore",
    message=".*can be slow with large amounts of data.*",
)

def qcReport(ds, dsInput, qc_rws_range):
    """wrapper method to make qc figures"""
    wsqc_fig = windSpeedQCfig(ds, qc_rws_range)
    scanqc_fig = scanFig(ds)
    angscat_fig, _ = angScatter(dsInput, ds)
    anghist_fig = angHistFig(ds, dsInput)
    
    return wsqc_fig, scanqc_fig, angscat_fig, anghist_fig

def windSpeedQCfig(ds, qc_rws_range):
    """wrapper method to make qc and probability scatter plots"""
    fig = plt.figure(figsize=(10, 4), layout="constrained")
    gs = GridSpec(nrows=1, ncols=3, width_ratios=[6, 0.25, 6], figure=fig)

    ax0 = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    ax1 = fig.add_subplot(gs[0, 2])

    probabilityScatter(ds, ax=ax0, fig=fig, cax=cax)
    probabilityVSrws(ds, qc_rws_range, ax=ax1, fig=fig)

    fig.suptitle(
        titleGenerator(ds, "QC of data", components=["location", "date", "file"])
    )

    return fig

def scanFig(ds):
    """Plot raw (top) and QC'd (bottom) radial wind speed vs time and range,
    with a vertical line marking the start of each scan-ID cycle (skipped for stare scans)."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(18, 10), sharex=True, sharey=True, layout="constrained"
    )

    stacked = ds[["wind_speed", "qc_wind_speed", "time"]].stack(obs=("scanID", "beamID"))
    stacked = stacked.isel(obs=~np.isnat(stacked.time.values))
    stacked = stacked.isel(obs=np.argsort(stacked.time.values))

    time = stacked.time.values
    range_vals = ds.range.values
    wind_speed_qc = stacked.wind_speed.where(stacked.qc_wind_speed == 0.0)

    im1 = ax1.pcolormesh(
        time, range_vals, stacked.wind_speed.values, cmap="coolwarm", vmin=-15, vmax=15, shading="auto"
    )
    im2 = ax2.pcolormesh(
        time, range_vals, wind_speed_qc.values, cmap="coolwarm", vmin=-15, vmax=15, shading="auto"
    )

    ax1.set_title("Raw")
    ax2.set_title("Quality-controlled")
    for ax, im in [(ax1, im1), (ax2, im2)]:
        ax.set_ylabel("Range [m]")
        fig.colorbar(im, ax=ax, label="Radial wind \n"+r"speed [m s$^{-1}$]")

    ax2.set_xlabel("Time (UTC)")
    utilities.format_time_xticks(ax2)

    if ds.attrs["scan_mode"].lower() != "stare":
        scan_starts = ds.time.min(dim="beamID").values
        scan_starts = scan_starts[~np.isnat(scan_starts)]
        for ax in (ax1, ax2):
            for t in scan_starts:
                ax.axvline(t, color="k", linewidth=0.75, alpha=0.6)

    fig.suptitle(
        titleGenerator(ds, "QC of data", components=["location", "date", "file"])
    )
    return fig

def angHistFig(ds, dsInput):
    """wrapper method to make angle histograms"""
    
    if ds.attrs['scan_mode'].lower()=='ppi':
        fig, ax = plt.subplots(1,1, figsize=(10, 6))
        fig, _ = anghist_1D(dsInput,ds,'azimuth', ax=ax, fig=fig)
        # fig, _ = angdiffhist_1D(dsInput, ds, 'azimuth',ax=ax[1], fig=fig,color='k')
    elif  ds.attrs['scan_mode'].lower()=='rhi':
        fig, ax = plt.subplots(1,1, figsize=(10, 6))
        fig, _ = anghist_1D(dsInput,ds,'elevation', ax=ax, fig=fig)
        # fig, _ = angdiffhist_1D(dsInput, ds,'elevation', ax=ax[1], fig=fig,color='k')
    elif  ds.attrs['scan_mode'].lower()=='3d':
        fig, ax = plt.subplots(1,1, figsize=(10, 6))
        fig, _ = anghist_2D(dsInput,ds, ax=ax, fig=fig)
        # fig, _ = angdiffhist_1D(dsInput, ds,'azimuth', ax=ax[1], fig=fig,color='k')
        # fig, _ = angdiffhist_1D(dsInput, ds,'elevation', ax=ax[2], fig=fig,color='k')
    elif ds.attrs['scan_mode'].lower()=='stare':
        return
        
    fig.suptitle(
        titleGenerator(ds, "Geometry standardization", components=["location", "date", "file"])
    )
    return fig

def probabilityScatter(ds, ax=None, cax=None, fig=None, **kwargs):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (9, 8)))

    prob_arange = np.arange(
        np.log10(ds.qc_wind_speed.attrs["min_probability_range"]+10**-10) - 1,
        np.log10(ds.qc_wind_speed.attrs["max_probability_range"]+10**-10) + 1,
    )

    sp = ax.scatter(
        ds.rws_norm,
        ds.snr_norm,
        s=3,
        c=np.log10(ds.probability),
        cmap=kwargs.get("cmap", "hot"),
        vmin=prob_arange[0],
        vmax=prob_arange[-1],
    )
    ax.set_xlabel(r"Normalized radial wind speed [m s$^{-1}$]")
    ax.set_ylabel("Normalized SNR [dB]")
    ax.grid(True)

    if cax is None:
        cax = fig.add_axes(
            [
                ax.get_position().x0 + ax.get_position().width + 0.01,
                ax.get_position().y0,
                0.015,
                ax.get_position().height,
            ]
        )
    cbar = plt.colorbar(sp, cax=cax, label="Probability")
    cbar.set_ticks(prob_arange)
    cbar.set_ticklabels([r"$10^{" + str(int(p)) + "}$" for p in prob_arange])
    # fig.tight_layout()
    return ax


# Second function: Semilog plot
def probabilityVSrws(ds, qc_rws_range, ax=None, fig=None, **kwargs):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (9, 8)))

    ax.semilogx(
        ds.probability.values.ravel(),
        ds.rws_norm.values.ravel(),
        '.',
        color='C0',
        alpha=kwargs.get("alpha", 0.1),
    )
    ax.semilogx(
        0,
        0,
        '.',
        color='C0',
        label=kwargs.get("label_all", "All points"),
    )
    ax.semilogx(
        [10**b.mid for b in qc_rws_range.index],
        qc_rws_range.values,
        "-^",
        color='C1',
        markersize=kwargs.get("markersize", 5),
        label=kwargs.get("label_range", "Range"),
    )
    ax.semilogx(
        [
            ds["qc_wind_speed"].attrs["qc_probability_threshold"],
            ds["qc_wind_speed"].attrs["qc_probability_threshold"],
        ],
        [
            -ds["qc_wind_speed"].attrs["rws_norm_limit"],
            ds["qc_wind_speed"].attrs["rws_norm_limit"],
        ],
        "--",
        color='C2',
        label=kwargs.get("label_threshold", "Threshold"),
    )

    ax.set_xlabel("Probability")
    ax.set_ylabel(r"Normalized radial wind speed [m s$^{-1}$]")
    ax.yaxis.set_label_position("right")
    ax.legend()
    ax.grid(True)

    return ax

def angScatter(dsInput, dsStandardized, ax=None, fig=None, **kwargs):
    """Scatter plot showing observed and standardized azimuth and elevation angles"""
    if fig is None or ax is None:
        fig, ax = plt.subplots(2,1,figsize=kwargs.get("figsize", (16, 8)))

    #azimuth time series
    ax[0].scatter(
        dsInput.time,
        dsInput.azimuth,
        marker=".",
        c="C0",
        label="Input data",
    )
    ax[0].scatter(
        dsStandardized.time,
        dsStandardized.azimuth,
        marker=".",
        edgecolor="C1",
        facecolor='none',
        label="Standardized data",
    )
 
    xformatter = mdates.DateFormatter("%H:%M")
    ax[0].xaxis.set_major_formatter(xformatter)
    ax[0].set_xlabel("Time (UTC)")
    ax[0].set_ylabel("Azimuth angle [˚]")
 
    ax[0].set_title(
        titleGenerator(
            dsStandardized,
            "Azimuth angles",
            ["location", "date", "file"],
        )
    )
    ax[0].legend()
    ax[0].grid(True)
    
    #elevation time series
    ax[1].scatter(
        dsInput.time,
        dsInput.elevation,
        marker=".",
        c="C0",
        label="Input data",
    )
    ax[1].scatter(
        dsStandardized.time,
        dsStandardized.elevation,
        marker=".",
        edgecolor="C1",
        facecolor='none',
        label="Standardized data",
    )

    xformatter = mdates.DateFormatter("%H:%M")
    ax[1].xaxis.set_major_formatter(xformatter)
    ax[1].set_xlabel("Time (UTC)")
    ax[1].set_ylabel("Elevation angle [˚]")

    ax[1].set_title(
        titleGenerator(
            dsStandardized,
            "Elevation angles",
            ["location", "date", "file"],
        )
    )
    ax[1].legend()
    ax[1].grid(True)
    
    fig.tight_layout()
    
    return fig, ax

def anghist_1D(dsInput,ds,which_angle='azimuth',ax=None, fig=None, **kwargs):
    """Bar plot of occurrences of angles"""
    
    assert which_angle in ['azimuth','elevation'], f'{which_angle} is not a valid beam angle'
        
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (5, 3)))
    
    
    ang=dsInput[which_angle].values.flatten()
    counts, bins = np.histogram(ang, bins=utilities.rev_mid(np.unique(ang)))
    ax.bar(np.unique(ang),counts, width=0.5,color='C0',alpha=0.75,label='Input data', **kwargs)

    ang=ds[which_angle].values.flatten()
    counts, bins = np.histogram(ang, bins=utilities.rev_mid(np.unique(ang)))
    ax.bar(np.unique(ang),counts, width=0.5,alpha=1,edgecolor='C1',facecolor='none',label='Standardized data', **kwargs)
    ax.set_xlabel(f"{which_angle.capitalize()} angle [˚]")
    ax.set_ylabel("Occurrence")
    ax.legend()
    ax.grid(True)

    return fig, ax

def anghist_2D(dsInput,ds,which_angle='azimuth',ax=None, fig=None, **kwargs):
    """2D trajectory of scans colored by occurrence"""
        
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (5, 3)))
    
    azi=dsInput['azimuth'].values.flatten()
    ele=dsInput['elevation'].values.flatten()
    ax.plot(azi,ele,'.',color='C0',markersize=12,alpha=0.5,label='Input data', **kwargs)
    
    azi=ds['azimuth'].mean(dim='scanID').values.flatten()
    ele=ds['elevation'].mean(dim='scanID').values.flatten()
    
    ax.plot(azi,ele,'--.',color='C1',markersize=10,label='Standardized data', **kwargs)
    ax.set_xlabel(r'Azimuth [$^\circ$]')
    ax.set_ylabel(r'Elevation [$^\circ$]')
    ax.legend()
    ax.grid(True)

    return fig, ax

def titleGenerator(ds, inputString: str = None, components=["location", "date"]):
    """Build title for figures"""
    componentDict = {
        "date": f"on {ds.time.dt.strftime('%Y-%m-%d').values.flatten()[0]}",
        "location": f"at {ds.attrs['location_id']}",
        "file": f"\nFile: {ds.attrs['input_source']}",
    }

    title = [inputString] + [componentDict[comp] for comp in components]

    return " ".join(title)