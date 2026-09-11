# Helper functions for ML Mapmaking

import re
import numpy as np


from astropy import units as u
import matplotlib.pyplot as plt

import toast
from toast.observation import default_values as defaults



def group_dets(obs, mask=defaults.det_mask_nonscience):
    """Organize timestreams associated with detectors."""
    pat = re.compile(r"(demod[024ri]+)_(.*)")
    fp = obs.telescope.focalplane.detector_data
    if "pixel" in fp.colnames:
        det_to_pix = dict()
        for row in fp:
            det_to_pix[row["name"]] = row["pixel"]
    else:
        det_to_pix = {x: x for x in fp["name"]}
    groups = dict()
    
    for det in obs.select_local_detectors(flagmask=mask):
        pix = det_to_pix[det]
        if pix not in groups:
            groups[pix] = dict()
        mat = pat.match(det)
        if mat is None:
            # We have normal, undemodulated data
            det_type = "normal"
            det_name = det
        else:
            # This is a demod TOD
            det_type = mat.group(1)
            det_name = mat.group(2)
        if det_name not in groups[pix]:
            groups[pix][det_name] = dict()
            groups[pix][det_name]["normal"] = det_name
        groups[pix][det_name][det_type] = det
    return groups

def plot_obs_dets(
    obs,
    d_start=0,
    d_end=None,
    s_start=0,
    s_end=None,
    view=None,
    signal=defaults.det_data,
    mask=defaults.det_mask_nonscience,
    file=None,
    pattern=None,
):
    """Plot some unflagged detectors in an observation.

    Args:
        obs (Observation):  The observation
        d_start (int):  The starting local detector index to plot.
        d_end (int): The local detector index limit to plot.
        s_start (int):  The starting sample index to plot.
        s_end (int):  The sample index limit to plot
        view (str):  The optional intervals to overplot.
        signal (str):  The detdata name to plot.
        mask (int):  Mask for selecting good dets.
        file (str):  If not None, save to file instead.
        pattern (str):  Regex pattern to select detector IDs

    """
    if s_start is None:
        s_start = 0
    if s_end is None:
        s_end = obs.n_local_samples
    slc = slice(s_start, s_end, 1)

    # Interval view
    rects = None
    if view is not None and view in obs.intervals:
        rects = list()
        for intr in obs.intervals[view]:
            begin = intr.first
            end = intr.last
            if begin < s_end and end > s_start:
                # Some overlap
                if begin < s_start:
                    begin = s_start
                if end > s_end:
                    end = s_end
                rects.append((begin, end))

    # Get valid dets
    groups = group_dets(obs, mask=mask)
    n_all_dets = np.sum([len(y) for x, y in groups.items()])

    dets = dict()
    if d_start is None:
        d_start = 0
    if d_end is None:
        d_end = n_all_dets
    cur = 0
    for grp, gdets in groups.items():
        ngdet = len(gdets)
        if cur + ngdet > d_start and cur < d_end:
            dets.update(gdets)
        cur += ngdet

    if pattern is not None:
        det_pat = re.compile(pattern)
    else:
        det_pat = re.compile(r".*")
    fp = obs.telescope.focalplane.detector_data
    fpvals = {x: y for x, y in zip(fp["det_info:readout_id"], fp["det_info:det_id"])}

    # Compute number of plots
    n_plot = 1
    for d, dtod in dets.items():
        if "demod2r" in dtod:
            # We have demodulated data with 2f component
            n_plot = 4
            demod = True
            demod2f = True
        elif "demod0" in dtod:
            # We have normal 0 and 4f demodulated data
            n_plot = 3
            demod = True
            demod2f = False
        else:
            # Normal data
            demod = False
            demod2f = False

    # Extra plot for flags
    n_plot += 1

    fig, axs = plt.subplots(nrows=n_plot, ncols=1, dpi=100, figsize=(12, 8 * n_plot))

    # Shared flags
    axs[-1].plot(
        obs.shared[defaults.times].data[slc],
        obs.shared[defaults.shared_flags].data[slc],
        "-",
        color="black",
        label="Shared Flags",
    )

    for idet, (det, dtod) in enumerate(dets.items()):
        if idet < d_start or idet >= d_end:
            continue
        iplot = 0
        if demod:
            if det_pat.match(fpvals[det]) is None:
                continue
            # Plot demod0
            axs[iplot].plot(
                obs.shared[defaults.times].data[slc],
                obs.detdata[signal][dtod["demod0"], slc],
                "-",
                label=det,
            )
            iplot += 1
            if demod2f:
                # Plot 2f magnitude
                d2r = dtod["demod2r"]
                d2i = dtod["demod2i"]
                d2data = np.sqrt(
                    obs.detdata[signal][d2r, slc] ** 2 +
                    obs.detdata[signal][d2i, slc] ** 2
                )
                axs[iplot].plot(
                    obs.shared[defaults.times].data[slc],
                    d2data,
                    "-",
                    label=det,
                )
                iplot += 1
            # Plot 4f
            axs[iplot].plot(
                obs.shared[defaults.times].data[slc],
                obs.detdata[signal][dtod["demod4r"], slc],
                "-",
                label=det,
            )
            iplot += 1
            axs[iplot].plot(
                obs.shared[defaults.times].data[slc],
                obs.detdata[signal][dtod["demod4i"], slc],
                "-",
                label=det,
            )
            iplot += 1
            axs[iplot].plot(
                obs.shared[defaults.times].data[slc],
                obs.detdata[defaults.det_flags][dtod["demod0"], slc],
                "-",
                label=det,
            )
        else:
            if det_pat.match(fpvals[det]) is None:
                continue
            # Plot normal TOD
            axs[iplot].plot(
                obs.shared[defaults.times].data[slc],
                obs.detdata[signal][dtod["normal"], slc],
                "-",
                label=det,
            )
            iplot += 1
            axs[iplot].plot(
                obs.shared[defaults.times].data[slc],
                obs.detdata[defaults.det_flags][dtod["normal"], slc],
                "-",
                label=det,
            )
            iplot += 1

    # Views and Labels
    trects = None
    if rects is not None:
        trects = [
            (obs.shared[defaults.times].data[x], obs.shared[defaults.times].data[y-1])
            for x, y in rects
        ]

    def _plot_rects(ax):
        if trects is None:
            return
        ymin, ymax = ax.get_ylim()
        for rct in trects:
            ax.add_patch(
                mpatches.Rectangle(
                    (rct[0], ymin), 
                    rct[1]-rct[0], 
                    ymax-ymin,
                    fill=True,
                    facecolor="gray",
                    alpha=0.2,
                )
            )

    iplot = 0
    if demod:
        _plot_rects(axs[iplot])
        axs[iplot].legend(loc="best")
        axs[iplot].set_title("Demodulated Intensity")
        iplot += 1
        if demod2f:
            _plot_rects(axs[iplot])
            axs[iplot].legend(loc="best")
            axs[iplot].set_title(r"Demodulated 2f Magnitude ($\sqrt{{demod2r}^2 + {demod2i}^2}$)")
            iplot += 1
        _plot_rects(axs[iplot])
        axs[iplot].legend(loc="best")
        axs[iplot].set_title("Demodulated Q")
        iplot += 1
        _plot_rects(axs[iplot])
        axs[iplot].legend(loc="best")
        axs[iplot].set_title("Demodulated U")
        iplot += 1
    else:
        _plot_rects(axs[iplot])
        axs[iplot].legend(loc="best")
        axs[iplot].set_title("Timestreams")
        iplot += 1
    _plot_rects(axs[iplot])
    axs[iplot].legend(loc="best")
    axs[iplot].set_title("Flags")
    iplot += 1

    if file is None:
        plt.show()
    else:
        fig.savefig(file)
    plt.close()