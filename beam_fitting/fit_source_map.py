import os, sys
import h5py
import numpy as np
import astropy.units as u


from pixell import enmap

# Add the LAT_beams repo to the path
path_to_repo = "/shared_home/henachman/software/LAT_beams"

if not os.path.exists(path_to_repo):
    raise ValueError(f"Path to repo does not exist: {path_to_repo}")

if path_to_repo not in sys.path:
    sys.path.append(path_to_repo)

import lat_beams.fitting.map.bessel as fb
import lat_beams.fitting.map.gauss as fg
from lat_beams.beam_utils import (
    crop_maps,
    estimate_cent,
    get_fwhm_radial_bins,
    process_model,
    radial_profile,
)
from lat_beams.fitting.map.base import make_guess
from lat_beams.plotting import plot_map_complete
from lat_beams.utils import (
    ErrCode,
    fail,
    get_args_cfg,
    init_log,
    set_tag,
    setup_cfg,
    setup_jobs,
    setup_paths,
    update_jobs_retry,
)
import argparse


def main(args):
    data_dir = args.data_dir
    # Starting at line 251 of the fit_source_map file
    map_path = os.path.join(
        data_dir, "mlmapmaker_sky_map.fits"
    )
    ivar_path = os.path.join(
        data_dir, "weights"
    )

    for i in range(1):
        # Load the maps
        try:
            solved = enmap.read_map(map_path)[0]
            solved = cast(enmap.ndmap, solved)
            weights = enmap.read_map(ivar_path)[0][0]
            weights = cast(enmap.ndmap, weights)
        except FileNotFoundError:
            msg = "Missing map files"
            to_save = (None, None)
            raise FileNotFoundError(msg)
        if solved.wcs is None:
            raise ValueError("WCS is None")
        pixsize = 3600 * solved.wcs.wcs.cdelt[1]  # type: ignore

        # Check if this is a bogus map
        if np.sum(~(weights == 0)) == 0:
            msg = "Weights all 0"
            to_save = (None, None)
            raise ValueError("This is a bogus map.")

        # Estimate SNR
        snr_extent_pix = int(cfg.snr_extent // pixsize)
        cent = estimate_cent(solved, weights, cfg.smooth_kern / pixsize, cfg.buf)
        sig = solved[cent]
        noise = solved.copy()
        xmin = max(0, cent[0] - snr_extent_pix)
        xmax = min(solved.shape[0], cent[0] + snr_extent_pix)
        ymin = max(0, cent[1] - snr_extent_pix)
        ymax = min(solved.shape[1], cent[1] + snr_extent_pix)
        noise[xmin:xmax, ymin:ymax] = np.nan
        noise = np.nanstd(np.diff(noise))
        snr = sig / noise

        if snr < cfg.min_snr:
            msg = "Data SNR too low"
            to_save = (None, None)
            raise ValueError(msg)

        # Slice things
        solved, weights = crop_maps([solved, weights], cent, int(cfg.extent // pixsize))
        posmap = enmap.posmap(solved.shape, solved.wcs)
        cent = estimate_cent(solved, weights, cfg.smooth_kern / pixsize, cfg.buf_cropped)
        fscale_fac = 90.0 / float(band[1:]) if cfg.apply_fscale else 1
        band_mask_size = np.deg2rad(fscale_fac * cfg.mask_size)

        # Make weights and zero things out
        weights[~np.isfinite(weights)] = 0
        weights[~np.isfinite(solved)] = 0
        solved[~np.isfinite(solved)] = 0

        # Setup aman for output
        aman = AxisManager()
        aman.wrap("noise", noise * u.pW)


        # Fit gaussian model
        cent, smoothed = estimate_cent(
            solved, weights, cfg.smooth_kern / pixsize, cfg.buf_cropped, True
        )
        maxval = np.max(solved)
        guess = make_guess(
            amp=np.nan_to_num(smoothed[cent].item(), True, maxval, maxval, maxval),
            fwhm_xi=np.deg2rad(cfg.nominal_fwhm[band] / 60.0),
            fwhm_eta=np.deg2rad(cfg.nominal_fwhm[band] / 60.0),
            xi0=posmap[1][cent[0], cent[1]].item(),
            eta0=posmap[0][cent[0], cent[1]].item(),
            phi=0,
            off=0,
        )
        gauss_params, model = fg.fit_gauss_map(
            solved,
            weights,
            posmap,
            guess,
            "pW",
            cfg.sym_gauss,
            -1,
        )
        if gauss_params is None or model is None:
            msg = "Gauss fit failed"
            to_save = (None, None)
            raise ValueError(msg)

        # Compute the gaussian model
        model = fg.gaussian2d_from_aman(posmap, gauss_params)

        # Check clipping
        c = np.unravel_index(np.argmax(model, axis=None), model.shape)
        min_c_dist = np.min(np.hstack((c, np.array(solved.shape) - np.array(c)))) * pixsize
        if min_c_dist < 120 * cfg.nominal_fwhm[band]:
            msg = "Source too close to edge of map"
            to_save = (None, None)
            raise ValueError(msg)

        # Get FWHM from data
        cent = (int(c[-1]), int(c[-2]))
        rprof = radial_profile(solved - gauss_params.off.value, cent)
        r = np.linspace(0, len(rprof), len(rprof)) * pixsize
        rmsk = r < 3 * 60 * cfg.nominal_fwhm[band] / 2.355
        data_fwhm = get_fwhm_radial_bins(r[rmsk], rprof[rmsk], interpolate=True) * u.arcsec
        aman.wrap("data_fwhm", data_fwhm)
        aman.wrap("r", r * u.arcsec)
        aman.wrap("rprof", rprof * u.pW)

        # FWHM check
        if (
            np.isnan(data_fwhm)
            or abs(1 - data_fwhm.value / (60 * cfg.nominal_fwhm[band])) > cfg.fwhm_tol
        ):
            msg = "Data FWHM out of tolerance"
            to_save = (None, None)
            raise ValueError(msg)

        # Process and save fit model
        gauss_params = process_model(
            gauss_params,
            solved - gauss_params.off.value,
            model - gauss_params.off.value,
            float(noise),
            cfg.min_snr,
            (cent[-2], cent[-1]),
            u.pW,
            pixsize,
            data_fwhm,
            cfg.min_sigma,
            job,
            logger,
        )
        if gauss_params is None:
            to_save = (None, None)
            continue
        aman.wrap("gauss", gauss_params)
        off = gauss_params.off.value
        for to_parent in ["amp", "off", "xi0", "eta0"]:
            aman.wrap(to_parent, gauss_params[to_parent])
        aman.wrap("final_model", "gauss")

        # Get bessel beam if we want
        if cfg.bessel_beam:
            try:
                bessel_beam_params, model = fb.fit_bessel_map(
                    solved,
                    weights,
                    posmap,
                    gauss_params,
                    "pW",
                    cfg.n_bessel,
                    cfg.n_multipoles,
                    cfg.aperature,
                    const.c / (float(band[1:]) * u.GHz),  # type: ignore
                    band_mask_size,
                    np.log((10**cfg.bessel_wing_n_sigma) / fscale_fac) / np.log(10),
                    cfg.skip_multipoles,
                )
            except Exception as e:
                msg = f"Bessel fit failed with error {str(e)}"
                to_save = (None, None)
                raise Exception(msg)
            if bessel_beam_params is None or model is None:
                msg = "Bessel fit failed"
                to_save = (None, None)
                raise ValueError(msg)

            off = bessel_beam_params.off.value
            bessel_beam_params = process_model(
                bessel_beam_params,
                solved - bessel_beam_params.off.value,
                model - bessel_beam_params.off.value,
                float(noise),
                cfg.min_snr,
                (cent[-2], cent[-1]),
                u.pW,
                pixsize,
                data_fwhm,
                cfg.min_sigma,
                job,
                logger,
            )
            if bessel_beam_params is None:
                to_save = (None, None)
                raise ValueError("Bessel beam params is None")
            aman.wrap("bessel", bessel_beam_params)
            aman.final_model = "bessel"


    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a WCS map from a FITS file.")
    parser.add_argument(
        "data_dir",
        type=str,
        help="Directory containing the mlmapmaker_sky_map output files.",
    )

    args = parser.parse_args()

    main(args)