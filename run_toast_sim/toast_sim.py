# Optionally change logging level
import os
os.environ["TOAST_LOGLEVEL"] = "INFO"
# This is needed before importing toast, and should
# match the value passed to the '-t' option of %toast
# os.environ["OMP_NUM_THREADS"] = "8"

# Imports

import numpy as np
import astropy.units as u
from astropy.table import QTable

# If running offline astropy will complain about not being able to find the IERS data.
# This blocks the warning (but does not fix the problem)
from astropy.utils.iers import conf
conf.auto_max_age = None


import pandas as pd
import os
import sys
from time import time
import pickle
import shutil
from datetime import datetime

import toast
from toast.tests import helpers
from toast.observation import default_values as defaults
from toast.vis import plot_projected_quats

import inspect
import healpy as hp
import h5py
from time import time

from toast import Telescope, Data
from toast.instrument import Focalplane, GroundSite
from toast.instrument_sim import (
    fake_rhombihex_focalplane,
    fake_boresight_focalplane,
    fake_hexagon_focalplane,
    plot_focalplane
)

# Add the path to my custom operators here
abspath = os.path.dirname(os.path.abspath(__file__))
repo_path = abspath.split("SO_PointingReqs")[0]
ops_path = os.path.join(repo_path, "SO_PointingReqs")
if ops_path not in os.sys.path:
    os.sys.path.append(ops_path)

from custom_toast_ops import PointingOffset, PointingJitter
from run_toast_sim.pointing_vis import boresight_pointing_3dvis, boresight_pointing_residualplot
from run_toast_sim.toast_mapmaker import toast_mapmaker, plot_maps
from run_toast_sim.plotting_utils import plot_dets, plot_scanning
# Start the timer
time_start = time()
dt_timestart = datetime.fromtimestamp(time_start)

# MPI communicator
world, procs, rank = toast.mpi.get_world()
comm = helpers.create_comm(world, single_group=True)

env = toast.Environment.get()
print(env)

def op_from_file(file_path):
    """
    Load a TOAST Operator from a pickle file.
    
    Parameters:
    -----------
    file_path : str
        Path to the pickle file containing the serialized TOAST Operator.

    Returns:
    --------
    toast.ops.Operator
        The loaded TOAST Operator.
    """


    with open(file_path, 'rb') as f:
        operator = pickle.load(f)
    
    return operator

def load_from_hdf5(file_path, class_type):

    """
    Load a TOAST object (like a Telescope, Focalplane, or GroundSite) from an HDF5 file.
    """
    
    with h5py.File(file_path, 'r') as f:
        group = f[list(f.keys())[0]]
        obj = class_type.load_hdf5(group)
    
    return obj

def main(args, log_file_path=None):
    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # MARK: Make a Telescope
    if args.load_telescope is not None:
        # if the telescope hdf5 file is provided, load it
        telescope = load_from_hdf5(args.load_telescope, Telescope)
    else:
        # Need to create a telescope.
        # Check if a site file has been provided.
        if args.load_site is not None:
            site = load_from_hdf5(args.load_site, GroundSite)
        else:
            # Create a site
            print(f"Site file not specified, creating a new site with the following parameters:\n  {args.site_name=}\n  {args.site_lat=}\n  {args.site_lon=}\n  {args.site_alt=}")
            site = GroundSite(args.site_name, args.site_lat, args.site_lon, args.site_alt * u.meter)
            if args.save_intermediate:
                # Save the site to an hdf5 file for later use
                site_save_path = os.path.join(out_dir, "site.hdf5")
                with h5py.File(site_save_path, 'w') as f:
                    group = f.create_group("site")
                    site.save_hdf5(group)
                print(f"Site saved to {site_save_path}")

        if args.load_focalplane is not None:
            # load the focal plane from the provided hdf5 file
            focalplane = load_from_hdf5(args.load_focalplane, Focalplane)
        else:
            # Create a site
            print(f"Focal Plane file not specified, creating a new focal plane with the following parameters:\n  {args.fov=}\n  {args.pix_num=}\n  {args.fwhm=}\n  {args.band_center=}\n  {args.band_width=}")
            fp_fwhm = args.fwhm * u.arcmin

            focalplane = fake_rhombihex_focalplane(
                n_pix_rhombus=args.pix_num,# Number of pixels per rhombus must be a perfect square
                width=args.fov * u.degree, # field of view of the focal plane
                gap=0 * u.radian,
                sample_rate=10.0 * u.Hz,
                epsilon=0.0,
                fwhm=fp_fwhm, # Beam size
                bandcenter=args.band_center * u.GHz,
                bandwidth=args.band_width * u.GHz,
                psd_net=300.0 * u.uK * np.sqrt(1 * u.second),
                psd_fmin=1.0e-5 * u.Hz,
                psd_alpha=1.0,
                psd_fknee=0.05 * u.Hz,
                fwhm_sigma=0.0 * u.arcmin,
                bandcenter_sigma=0 * u.GHz,
                bandwidth_sigma=0 * u.GHz,
                random_seed=123456,
            )
            if args.save_intermediate:
                # Save the focal plane to an hdf5 file for later use
                focalplane_save_path = os.path.join(out_dir, "focalplane.hdf5")
                with h5py.File(focalplane_save_path, 'w') as f:
                    group = f.create_group("focalplane")
                    focalplane.save_hdf5(group)
                print(f"Focal plane saved to {focalplane_save_path}")

        telescope = toast.Telescope(name=args.telescope_name, focalplane = focalplane, site = site)
        if args.save_intermediate:
            # Save the telescope to an hdf5 file for later use
            telescope_save_path = os.path.join(out_dir, "telescope.hdf5")
            with h5py.File(telescope_save_path, 'w') as f:
                group = f.create_group("telescope")
                telescope.save_hdf5(group)
            print(f"Telescope saved to {telescope_save_path}")
    ck1 = time()
    print(f"Time to load telescope {ck1 - time_start:.2f} seconds")

    data = toast.Data(comm)
    schedule = toast.schedule.GroundSchedule()
    schedule.read(args.schedule_file)

    # Populate observations according to the schedule and telescope.
    sim_ground = toast.ops.SimGround(
        telescope=telescope,
        weather="atacama",
        detset_key="pixel",
        schedule=schedule,
        median_weather=True, # no longer random weather, but less chance of an outlier
    )

    sim_ground.apply(data)

    # MARK: Pointing
    det_point_azel = toast.ops.PointingDetectorSimple(
        boresight=defaults.boresight_azel,
        quats="quats_azel"
    )
    det_point_radec = toast.ops.PointingDetectorSimple(
        boresight=defaults.boresight_radec,
        quats="quats_radec"
    )
    # Pixelization. 

    # Load the input map and match the nside to the map.
    load_input_map = hp.read_map(args.input_map)

    nside = hp.get_nside(load_input_map)
    pixels_radec = toast.ops.PixelsHealpix(
        nside=nside,
        nest=True,
        detector_pointing=det_point_radec,
    )
    # Stokes weights.  This just uses focalplane table properties to treat each detector
    # as a linear polarizer with possibly some cross-polar response.

    weights_azel = toast.ops.StokesWeights(
        mode="IQU",
        detector_pointing=det_point_azel,
    )

    weights_radec = toast.ops.StokesWeights(
        mode="IQU",
        detector_pointing=det_point_radec,
    )

    pixels_radec.apply(data)
    weights_radec.apply(data)


    # Pixel Distribution
    pix_dist = toast.ops.BuildPixelDistribution(
        pixel_dist="pixel_dist",
        pixel_pointing=pixels_radec,
    )
    pix_dist.apply(data)

    init_boreradec = np.array(data.obs[0].shared[defaults.boresight_radec].data)

    if args.pointing_offset:
        # Code to apply a constant pointing offset
        pointing_offset = PointingOffset(
            dxi=args.dxi,  # Deflection in Xi coordinates (in radians)
            deta=args.deta,  # Deflection in Eta coordinates (in radians)
            boresight_azel=defaults.boresight_azel,
            boresight_radec=defaults.boresight_radec,
        )
        pointing_offset.apply(data)

    if args.pointing_jitter:
        # Code to apply a random pointing jitter
        pointing_jitter = PointingJitter(
            max_daz=args.max_daz,  # Maximum deflection in Azimuth coordinates (in radians)
            max_del=args.max_del,  # Maximum deflection in Elevation coordinates (in radians)
            corotator_max=args.corotator_max,  # Maximum deflection in the corotator angle (in radians)
            sin_amp=args.sin_amp,  # Amplitude of the sinusoidal jitter (in radians)
            sin_freq=args.sin_freq,  # Frequency of the sinusoidal jitter (in 1/samples)
            boresight_azel=defaults.boresight_azel,
            boresight_radec=defaults.boresight_radec,
        )
        pointing_jitter.apply(data)
    


    # Pointing vis
    ob = data.obs[0]

    jitter_boreradec = np.array(ob.shared[defaults.boresight_radec].data)
    boresight_pointing_3dvis(jitter_boreradec, init_boreradec, ob, n_slices=30, step_size=10, rad_threshold=0.15, show_dets=True, out_dir=out_dir)
    boresight_pointing_residualplot(jitter_boreradec, init_boreradec, n_slices=30, step_size=10, out_dir=out_dir)

    ck2 = time()
    print(f"Time to apply pointing {ck2 - ck1:.2f} seconds")

    # MARK: Scan Sky
    # Scan the map

    scan_map = toast.ops.ScanHealpixMap(
        file=args.input_map,
        pixel_pointing=pixels_radec,
        stokes_weights=weights_radec,
    )
    scan_map.apply(data)

    # Checkpoint : Save a plot of some detector data to make sure that we are actually scanning the map correctly.
    if rank == 0:
        # only plot a few detectors
        plot_scanning(ob, s_start=0, s_end=5000, out_dir=out_dir)
        plot_dets(ob, d_start=0, d_end=5, s_start=0, s_end=5000, view="scanning", out_dir=out_dir, file_name="post_scan_")

    ck3 = time()
    print(f"Time to scan map {ck3 - ck2:.2f} seconds")

    if not args.no_noise:
        # MARK: Instrument Noise
        if args.nominal_noise is not None:
            # load the nominal noise from disk
            nominal_noise = op_from_file(args.nominal_noise)
        else:
            nominal_noise = toast.ops.DefaultNoiseModel()
        nominal_noise.apply(data)

        # MARK: Sim noise
        if args.sim_noise is not None:
            # load the sim noise from disk
            sim_noise = op_from_file(args.sim_noise)
        else:
            sim_noise = toast.ops.SimNoise(
                noise_model=nominal_noise.noise_model,
            )
        sim_noise.apply(data)

    ck4 = time()
    print(f"Time to apply noise {ck4 - ck3:.2f} seconds")
    # MARK: Scan Sync Signal
    if not args.no_scan_sync:
        ground_pickup = toast.ops.SimScanSynchronousSignal(
            detector_pointing=det_point_azel,
            scale=0.001 * u.K,
            stokes_weights=weights_azel,
        )
        ground_pickup.apply(data)

    # MARK: Sim Atm
    if not args.no_atmosphere:
        sim_atm = toast.ops.SimAtmosphere(
            detector_pointing=det_point_azel,
            add_loading=True,
            lmin_center=0.001 * u.m,
            lmin_sigma=0.0001 * u.m,
            lmax_center=1.0 * u.m,
            lmax_sigma=0.1 * u.m,
            xstep=20 * u.m,
            ystep=20 * u.m,
            zstep=20 * u.m,
            zmax=200 * u.m,
            gain=4e-5,
            wind_dist=1000 * u.m,
        )
        sim_atm.apply(data)


    # Checkpoint : Save a plot of some detector data to make sure that we are actually scanning the map correctly.
    if rank == 0:
        plot_dets(ob, d_start=0, d_end=5, s_start=0, s_end=5000, view="scanning", out_dir=out_dir, file_name="post_atm_")

    ck5 = time()
    print(f"Time to apply atmosphere {ck5 - ck4:.2f} seconds")

    # MARK: Save Obs
    data_save_path = os.path.join(out_dir, "toast_obs")
    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)
    else:
        # Remove the existing directory and its contents
        shutil.rmtree(data_save_path)
        print(f"Clearing and replacing existing directory: {data_save_path}")
    save_data = toast.ops.SaveHDF5(volume=data_save_path)
    save_data.apply(data)


    # MARK: Mapmaking
    # if requested, run a simple filter bin mapmaker.
    if args.make_map:
        print("Running mapmaker...")
        toast_mapmaker(data=data, out_dir=os.path.join(out_dir, "mapmaker"))
    
    plot_maps(
        root=os.path.join(out_dir, "mapmaker"),
        gnom_res = 8.0,
        gnomrot=(290, -60),
        xsize = 2000,
        range_I=(-0.01, 0.01),
        range_Q=(-0.0002, 0.0002),
        range_U=(-0.0002, 0.0002),
        max_hits=1000,
        truth=None,  # You can specify a truth map file here if available
    )

    # END
    end_time = time()
    dt_timeend = datetime.fromtimestamp(end_time)
    if log_file_path:
        with open(log_file_path, "a") as log_file:
            log_file.write(f"Simulation run End Time : {dt_timeend}\n")
            log_file.write(f"Total time for simulation: {dt_timeend - dt_timestart}\n")
    return

# MARK: Argparse
# Run the main function if the script is executed directly
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run TOAST simulation")

    # Add required command line arguments
    parser.add_argument(
        "-i",
        "--input_map",
        type=str,
        required=True,
        help="Path to the input sky map file (FITS format)",
    )
    parser.add_argument(
        "-s",
        "--schedule_file",
        type=str,
        required=True,
        help="Path to the schedule file (required)",
    )
    # Add optional command line arguments for loading simulation configuration files
    parser.add_argument(
        "-t",
        "--load_telescope",
        type=str,
        default=None,
        help="Path to the Telescope hdf5 file (optional)",
    )
    parser.add_argument(
        "--load_focalplane",
        type=str,
        default=None,
        help="Path to the focal plane hdf5 file (optional)",
    )
    parser.add_argument(
        "--load_site",
        type=str,
        default=None,
        help="Path to the site hdf5 file (optional)",
    )
    # Optional CLI arguments for telescope and site parameters
    parser.add_argument(
        "--telescope_name",
        type=str,
        default="MyTelescope",
        help="Name of the telescope to be created (default: MyTelescope)",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        default=".",
        help="Output directory for simulation results",
    )
    # Site parameters
    parser.add_argument(
        "--site_name",
        type=str,
        default="Atacama",
        help="Name of the site (default: Atacama)",
    )
    parser.add_argument(
            "--site_lat",
            type=str,
            default="-22:57:30",
            help="Latitude of the site (default: -22:57:30)",
        )
    parser.add_argument(
        "--site_lon",
        type=str,
        default="-67:47:10",
        help="Longitude of the site (default: -67:47:10)",
    )
    parser.add_argument(
        "--site_alt",
        type=float,
        default=5200.0,
        help="Altitude of the site in meters (default: 5200.0)",
    )
    # Focal Plane parameters
    parser.add_argument(
        "--fov",
        type=float,
        default=1.3,
        help="Field of view of the focal plane in degrees (default: 10.0)",
    )
    parser.add_argument(
        "--pix_num",
        type=int,
        default=16,
        help="Number of pixels per rhombus (must be a perfect square, default: 25)",
    )
    parser.add_argument(
        "--fwhm",
        type=float,
        default=30.0,
        help="Full width at half maximum (FWHM) of the beam in arcminutes (default: 30.0)",
    )
    parser.add_argument(
        "--band_center",
        type=float,
        default=150.0,
        help="Center frequency of the band in GHz (default: 150.0)",
    )
    parser.add_argument(
        "--band_width",
        type=float,
        default=20.0,
        help="Bandwidth of the band in GHz (default: 20.0)",
    )
    parser.add_argument(
        "--nominal_noise",
        type=str,
        default=None,
        help="Path to the nominal noise pickle file (optional)",
    )
    parser.add_argument(
        "--sim_noise",
        type=str,
        default=None,
        help="Path to the sim noise pickle file (optional). Note: this overrides the nominal noise if provided.",
    )

    parser.add_argument(
        "--save_intermediate",
        action="store_true",
        help="Flag to save intermediate results (optional)",
    )

    parser.add_argument(
        "--no_noise",
        action="store_true",
        help="Flag to skip simulating noise (optional)",
    )
    parser.add_argument(
        "--no_scan_sync",
        action="store_true",
        help="Flag to skip simulating scan synchronous signal (optional)",
    )
    parser.add_argument(
        "--no_atmosphere",
        action="store_true",
        help="Flag to skip simulating the atmosphere (optional)",
    )
    parser.add_argument(
        "--pointing_offset",
        action="store_true",
        help="Flag to apply a pointing offset (optional)",
    )

    # Pointing offset parameters
    try:
        idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    except KeyError:
        # This means the script is not running as part of a SLURM array job, so we can set idx to 0 or any other default value.
        idx = 0
    dxi_params  = [0.0, 0.01, 0.02, 0.03, 0.0, 0.0, 0.0]
    deta_params = [0.0, 0.0, 0.0, 0.0, 0.01, 0.02, 0.03]
    parser.add_argument(
        "--dxi",
        type=float,
        nargs='+',
        default=[dxi_params[idx]],
        help="Deflection in Xi coordinates (in radians) for pointing offset (default: [0.0])",
    )
    parser.add_argument(
        "--deta",
        type=float,
        nargs='+',
        default=[deta_params[idx]],
        help="Deflection in Eta coordinates (in radians) for pointing offset (default: [0.0])",
    )
    parser.add_argument(
        "-j",
        "--pointing_jitter",
        action="store_true",
        help="Flag to apply a pointing jitter (optional)",
    )

    
    daz_params  = [0.0, 0.01, 0.02, 0.03]
    del_params = [0.0, 0.01, 0.02, 0.03]
    parser.add_argument(
        "--max_daz",
        type=float,
        default=daz_params[idx],
        help="Maximum deflection in Azimuth coordinates (in radians) for pointing jitter (default: 0.01). For reference: 10arcsec ~= 5e-5 rad",
    )
    parser.add_argument(
        "--max_del",
        type=float,
        default=del_params[idx],
        help="Maximum deflection in Elevation coordinates (in radians) for pointing jitter (default: 0.01). For reference: 10arcsec ~= 5e-5 rad",
    )
    parser.add_argument(
        "--corotator_max",
        type=float,
        default=((0 * u.arcsec).to(u.rad)).value,
        help="Maximum deflection in the corotator angle (in radians) for pointing jitter (default: 0.01). For reference: 10arcsec ~= 5e-5 rad",
    )
    parser.add_argument(
        "--sin_amp",
        type=float,
        default=((100 * u.arcsec).to(u.rad)).value,
        help="Amplitude of the sinusoidal jitter (in radians) for pointing jitter (default: 0.005). For reference: 100arcsec ~= 5e-4 rad",
    )
    parser.add_argument(
        "--sin_freq",
        type=float,
        default=0.1,
        help="Frequency of the sinusoidal jitter (in 1/samples) for pointing jitter (default: 0.1)",
    )

    parser.add_argument(
        "--make_map",
        action="store_true",
        help="Flag to run the mapmaker after simulation (optional)",
    )

    args = parser.parse_args()

    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Make a log file that records the command line arguments and the time of execution
    log_file_path = os.path.join(args.out_dir, "simulation_log.txt")
    with open(log_file_path, "w") as log_file:
        log_file.write(f"Simulation run Start Time : {dt_timestart}\n")
        log_file.write(f"Simulation Command :\n")
        # Write the arguments split by dashes
        log_file.write(f"python ")
        for arg in sys.argv:
            if arg.startswith("-"):
                log_file.write(f" \\\n{arg} ")
            else:
                log_file.write(f"{arg}")
        log_file.write("\n")

    main(args, log_file_path=log_file_path)