import matplotlib.pyplot as plt
from toast.observation import default_values as defaults
import astropy.units as u
import numpy as np

def plot_dets(obs, d_start=0, d_end=None, s_start=0, s_end=None, view=None, signal=defaults.det_data):
    """Plot some detectors in an observation.
    
    Args:
        obs (Observation):  The observation
        d_start (int):  The starting local detector index to plot.
        d_end (int): The local detector index limit to plot.
        s_start (int):  The starting sample index to plot.
        s_end (int):  The sample index limit to plot
        view (str):  The optional intervals to overplot.
        signal (str):  The detdata name to plot.
    """
    slc = slice(s_start, s_end, 1)

    fig = plt.figure(dpi=100, figsize=(18, 12))
    ax = fig.add_subplot(2, 1, 1, aspect="auto")
    plt.gca().set_prop_cycle(None)
    for idet, det in enumerate(obs.select_local_detectors(flagmask=defaults.det_mask_nonscience)):
        if idet < d_start:
            continue
        if d_end is not None and idet >= d_end:
            continue
        ax.plot(
            obs.shared[defaults.times].data[slc], 
            obs.detdata[signal][det, slc], 
            '-',
            label=det,
        )
    ax.legend(loc="best")
    
    ax = fig.add_subplot(2, 1, 2, aspect="auto")
    
    if view is not None:
        inview = np.zeros_like(obs.shared[defaults.shared_flags].data[slc])
        begin = [x.first for x in obs.intervals[view]]
        end = [x.last+1 for x in obs.intervals[view]]
        for b, e in zip(begin, end):
            inview[b:e] = 1
        ax.plot(
            obs.shared[defaults.times].data[slc], 
            inview, 
            '-',
            color="red",
            label=f"View {view}",
        )
    ax.plot(
        obs.shared[defaults.times].data[slc], 
        obs.shared[defaults.shared_flags].data[slc], 
        '-',
        color="black",
        label="Shared Flags",
    )
    
    plt.gca().set_prop_cycle(None)
    for idet, det in enumerate(obs.select_local_detectors(flagmask=defaults.det_mask_nonscience)):
        if idet < d_start:
            continue
        if d_end is not None and idet >= d_end:
            continue
        ax.plot(
            obs.shared[defaults.times].data[slc], 
            obs.detdata[defaults.det_flags][det, slc], 
            '-',
            label=det,
        )
    ax.legend(loc="best")    
    plt.show()
    plt.close()

def plot_scanning(obs, s_start=0, s_end=None):
    slc = slice(s_start, s_end, 1)
    times = obs.shared[defaults.times].data[slc]
    az = obs.shared[defaults.azimuth].data[slc]
    el = obs.shared[defaults.elevation].data[slc]
    
    fig = plt.figure(dpi=100, figsize=(18, 12))
    ax = fig.add_subplot(2, 1, 1, aspect="auto")
    ax.plot(times, az, label="Azimuth")
    ax.set_xlabel("Posix Timestamps")
    ax.set_ylabel("Azimuth")
    ax = fig.add_subplot(2, 1, 2, aspect="auto")
    ax.plot(times, el, label="Elevation")
    ax.set_xlabel("Posix Timestamps")
    ax.set_ylabel("Elevation")
    plt.show()
    plt.close()

def plot_noise_model(model, model_fit=None, d_start=0, d_end=None):
    fig = plt.figure(dpi=100, figsize=(18, 12))
    ax = fig.add_subplot(1, 1, 1)
    plt.gca().set_prop_cycle(None)
    plot_max = 0
    plot_min = 1e100
    for idet, det in enumerate(model.detectors):
        if idet < d_start:
            continue
        if d_end is not None and idet >= d_end:
            continue
        freq = model.freq(det).to_value(u.Hz)
        psd = model.psd(det).to_value(u.K**2 * u.s)
        plot_min = min(plot_min, np.amin(psd))
        plot_max = max(plot_max, np.amax(psd))
        ax.loglog(
            freq,
            psd,
            label=det,
        )
    if model_fit is not None:
        # Also plot the fit
        plt.gca().set_prop_cycle(None)
        for idet, det in enumerate(model.detectors):
            if idet < d_start:
                continue
            if d_end is not None and idet >= d_end:
                continue
            freq = model_fit.freq(det)
            psd = model_fit.psd(det)
            ax.loglog(
                freq.to_value(u.Hz),
                psd.to_value(u.K**2 * u.s),
                label=f"{det} Fit",
            )
    freq = model.freq(model.detectors[0])
    
    ax.set_xlim(freq[0].to_value(u.Hz), freq[-1].to_value(u.Hz))
    ax.set_ylim(0.9 * plot_min, 1.1 * plot_max)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [K$^2$ / Hz]")
    ax.legend(loc="best")
    plt.show()
    plt.close()